/**
 * @file      Estimator.cpp
 * @brief     Main VIO estimator implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "FeatureTracker.h"
#include "Initializer.h"
#include "IMUPreintegrator.h"
#include "Optimizer.h"
#include "Camera.h"
#include "Frame.h"
#include "Feature.h"
#include "MapPoint.h"
#include "ConfigUtils.h"
#include "Logger.h"
#include <fstream>
#include <iomanip>

namespace vio_360 {

Estimator::Estimator()
    : m_frame_id_counter(0)
    , m_tracking_state(TrackingState::NOT_INITIALIZED)
    , m_current_pose(Eigen::Matrix4f::Identity())
    , m_transform_from_last(Eigen::Matrix4f::Identity())
    , m_gravity(0, 0, -9.81f)
    , m_gyro_bias(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_scale(1.0)
    , m_imu_init_frame_idx(-1)
    , m_first_keyframe_time(-1.0)
    , m_last_keyframe_time(-1.0) {
    
    // Initialize camera
    const auto& config = ConfigUtils::GetInstance();
    m_camera = std::make_shared<Camera>(
        config.camera_width,
        config.camera_height
    );
    m_boundary_margin = config.camera_boundary_margin;
    
    // Initialize feature tracker with INITIALIZATION settings (will switch to tracking after init)
    m_feature_tracker = std::make_unique<FeatureTracker>(
        m_camera,
        config.initialization_max_features,
        config.initialization_min_distance,
        config.initialization_quality_level,
        config.camera_boundary_margin
    );
    // Set grid params for initialization
    m_feature_tracker->SetGridParams(
        config.initialization_grid_cols,
        config.initialization_grid_rows,
        config.initialization_max_features_per_grid
    );
    
    // Initialize monocular initializer
    m_initializer = std::make_unique<Initializer>();
    
    // Initialize IMU preintegrator
    m_imu_preintegrator = std::make_unique<IMUPreintegrator>();
    
    // Load initialization parameters from config
    m_init_window_size = config.initialization_window_size;
    m_tracking_window_size = config.tracking_window_size;
    m_window_size = m_init_window_size;  // Start with init window size
    m_min_parallax = config.initialization_min_parallax;
    m_frame_window.reserve(m_window_size);
}

Estimator::~Estimator() {
    // Cleanup if needed
}

Estimator::EstimationResult Estimator::ProcessFrame(const cv::Mat& image, double timestamp) {
    EstimationResult result;
    
    // Create new frame
    m_current_frame = CreateFrame(image, timestamp);
    
    // =========================================================================
    // State Machine: NOT_INITIALIZED -> VISUAL_ONLY -> VIO (without IMU data)
    // =========================================================================
    
    switch (m_tracking_state) {
    case TrackingState::NOT_INITIALIZED: {
        if (m_previous_frame) {
            result.num_tracked = TrackFeatures();
        } else {
            DetectFeatures();
        }
        
        m_frame_window.push_back(m_current_frame);
        
        if (static_cast<int>(m_frame_window.size()) > m_window_size) {
            m_frame_window.erase(m_frame_window.begin());
        }
        
        LOG_DEBUG("  Window size: {}/{}, trying init...", m_frame_window.size(), m_window_size);
        
        if (static_cast<int>(m_frame_window.size()) == m_window_size) {
            bool init_result = TryInitialize();
            
            if (init_result) {
                result.init_success = true;
                m_tracking_state = TrackingState::VISUAL_ONLY;
                
                // Switch to tracking window size (smaller than init window)
                m_window_size = m_tracking_window_size;
                LOG_INFO("Initialization SUCCESS! Switching to tracking window size: {}", m_window_size);
                
                auto first_frame = m_frame_window.front();
                auto last_frame = m_frame_window.back();
                
                first_frame->SetKeyframe(true);
                last_frame->SetKeyframe(true);
                
                m_first_keyframe_time = first_frame->GetTimestamp();
                m_last_keyframe_time = last_frame->GetTimestamp();
                
                m_keyframes.push_back(first_frame);
                m_keyframes.push_back(last_frame);
                m_last_keyframe = last_frame;
                
                m_current_pose = last_frame->GetTwb();
            } else {
                if (m_frame_window.size() >= 2) {
                    auto first_frame = m_frame_window.front();
                    auto last_frame = m_frame_window.back();
                    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
                    
                    if (parallax >= m_min_parallax) {
                        result.init_ready = true;
                    }
                }
            }
        }
        break;
    }
    
    case TrackingState::VISUAL_ONLY:
    case TrackingState::VIO: {
        // Already initialized - normal tracking (no IMU data available)
        int num_tracked = TrackFeatures();
        
        LinkMapPointsFromPreviousFrame();
        
        int valid_mp_count = 0;
        const auto& features = m_current_frame->GetFeatures();
        for (size_t j = 0; j < features.size(); ++j) {
            auto mp = m_current_frame->GetMapPoint(static_cast<int>(j));
            if (mp && !mp->IsBad()) {
                valid_mp_count++;
            }
        }
        
        // Use constant velocity model (no IMU data in this overload)
        if (m_transform_from_last.isIdentity(1e-6)) {
            m_current_frame->SetTwb(m_previous_frame->GetTwb());
        } else {
            Eigen::Matrix4f predicted_pose = m_previous_frame->GetTwb() * m_transform_from_last;
            m_current_frame->SetTwb(predicted_pose);
        }
        
        if (valid_mp_count >= 6) {
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            if (pnp_result.success) {
                if (m_last_keyframe && !m_current_frame->IsKeyframe()) {
                    m_current_frame->SetReferenceKeyframe(m_last_keyframe);
                }
                
                m_current_pose = m_current_frame->GetTwb();
                LOG_INFO("Frame {}: PnP {} in/{} out, cost {:.2f}->{:.2f}",
                         m_current_frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                         pnp_result.initial_cost, pnp_result.final_cost);
                
                result.success = true;
            } else {
                LOG_WARN("Frame {}: PnP failed", m_current_frame->GetFrameId());
                result.success = false;
            }
        } else {
            LOG_WARN("Frame {}: Not enough MapPoints ({}) for PnP", 
                     m_current_frame->GetFrameId(), valid_mp_count);
            result.success = false;
        }
        
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        if (ShouldCreateKeyframe()) {
            CreateKeyframe();
        }
        
        result.num_tracked = num_tracked;
        break;
    }
    }  // end switch
    
    // Update state
    result.pose = m_current_pose;
    result.num_features = m_current_frame->GetFeatureCount();
    m_all_frames.push_back(m_current_frame);
    m_previous_frame = m_current_frame;
    
    return result;
}

Estimator::EstimationResult Estimator::ProcessFrame(
    const cv::Mat& image, 
    double timestamp,
    const std::vector<IMUData>& imu_data
) {
    EstimationResult result;
    
    // Create new frame first
    m_current_frame = CreateFrame(image, timestamp);
    
    // Store IMU data for this frame (for interpolation-based init)
    if (!imu_data.empty()) {
        m_imu_data_per_frame[m_current_frame->GetFrameId()] = imu_data;
    }
    
    // Process IMU data and compute preintegration
    if (!imu_data.empty() && m_previous_frame) {
        ProcessIMU(imu_data);
    }
    
    // =========================================================================
    // State Machine: NOT_INITIALIZED -> VISUAL_ONLY -> VIO
    // =========================================================================
    
    switch (m_tracking_state) {
    case TrackingState::NOT_INITIALIZED: {
        // =================================================================
        // State: NOT_INITIALIZED - Accumulate frames for 2-frame VO init
        // =================================================================
        if (m_previous_frame) {
            result.num_tracked = TrackFeatures();
        } else {
            DetectFeatures();
        }
        
        // Add current frame to window
        m_frame_window.push_back(m_current_frame);
        
        // Maintain window size
        if (static_cast<int>(m_frame_window.size()) > m_window_size) {
            m_frame_window.erase(m_frame_window.begin());
        }
        
        // Try to initialize when window is full
        if (static_cast<int>(m_frame_window.size()) == m_window_size) {
            bool init_result = TryInitialize();
            
            if (init_result) {
                result.init_success = true;
                
                // === Transition: NOT_INITIALIZED -> VISUAL_ONLY ===
                m_tracking_state = TrackingState::VISUAL_ONLY;
                
                LOG_INFO("VO Initialization SUCCESS! Transition to VISUAL_ONLY state.");
                
                // Set first and last frames as keyframes
                auto first_frame = m_frame_window.front();
                auto last_frame = m_frame_window.back();
                
                first_frame->SetKeyframe(true);
                last_frame->SetKeyframe(true);
                
                // Record keyframe times for ORB-SLAM3 style IMU init
                m_first_keyframe_time = first_frame->GetTimestamp();
                m_last_keyframe_time = last_frame->GetTimestamp();
                
                // Compute preintegration from first to last keyframe
                if (!m_imu_since_last_keyframe.empty()) {
                    Eigen::Vector3f gyro_bias = first_frame->GetGyroBias();
                    Eigen::Vector3f accel_bias = first_frame->GetAccelBias();
                    m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
                    
                    double start_time = m_imu_since_last_keyframe.front().timestamp;
                    double end_time = m_imu_since_last_keyframe.back().timestamp;
                    
                    auto preint = m_imu_preintegrator->Preintegrate(
                        m_imu_since_last_keyframe, start_time, end_time
                    );
                    
                    if (preint) {
                        last_frame->SetIMUPreintegrationFromLastKeyframe(preint);
                        LOG_INFO("  IMU preintegration: kf {} -> kf {}: dt={:.3f}s",
                                 first_frame->GetFrameId(), last_frame->GetFrameId(),
                                 preint->dt_total);
                    }
                    m_imu_since_last_keyframe.clear();
                }
                
                m_keyframes.push_back(first_frame);
                m_keyframes.push_back(last_frame);
                m_last_keyframe = last_frame;
                
                m_current_pose = last_frame->GetTwb();
                
                // IMU initialization will be done in VISUAL_ONLY state after 10 KFs + 2 seconds
            } else {
                // Check if ready for initialization
                if (m_frame_window.size() >= 2) {
                    auto first_frame = m_frame_window.front();
                    auto last_frame = m_frame_window.back();
                    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
                    
                    if (parallax >= m_min_parallax) {
                        result.init_ready = true;
                    }
                }
            }
        }
        break;
    }
    
    case TrackingState::VISUAL_ONLY: {
        // =================================================================
        // State: VISUAL_ONLY - PnP tracking, time-based KF, wait for IMU init
        // ORB-SLAM3: No IMU prediction, just visual tracking
        // =================================================================
        int num_tracked = TrackFeatures();
        
        // Link MapPoints from previous frame
        LinkMapPointsFromPreviousFrame();
        
        // Count valid MapPoints
        int valid_mp_count = 0;
        const auto& features = m_current_frame->GetFeatures();
        for (size_t j = 0; j < features.size(); ++j) {
            auto mp = m_current_frame->GetMapPoint(static_cast<int>(j));
            if (mp && !mp->IsBad()) {
                valid_mp_count++;
            }
        }
        
        // Use constant velocity model for initial pose (NO IMU prediction)
        if (m_transform_from_last.isIdentity(1e-6)) {
            m_current_frame->SetTwb(m_previous_frame->GetTwb());
        } else {
            Eigen::Matrix4f predicted_pose = m_previous_frame->GetTwb() * m_transform_from_last;
            m_current_frame->SetTwb(predicted_pose);
        }
        
        // PnP pose estimation
        if (valid_mp_count >= 6) {
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            if (pnp_result.success) {
                if (m_last_keyframe && !m_current_frame->IsKeyframe()) {
                    m_current_frame->SetReferenceKeyframe(m_last_keyframe);
                }
                
                m_current_pose = m_current_frame->GetTwb();
                m_transform_from_last = m_previous_frame->GetTwb().inverse() * m_current_pose;
                result.success = true;
            } else {
                LOG_WARN("Frame {}: PnP failed in VISUAL_ONLY", m_current_frame->GetFrameId());
                result.success = false;
            }
        } else {
            LOG_WARN("Frame {}: Not enough MapPoints ({}) for PnP", 
                     m_current_frame->GetFrameId(), valid_mp_count);
            result.success = false;
        }
        
        // Detect new features if needed
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        // ORB-SLAM3 style: Time-based keyframe creation (every 0.25s before IMU init)
        bool should_create_kf = false;
        double time_since_last_kf = timestamp - m_last_keyframe_time;
        constexpr double KF_INTERVAL_BEFORE_IMU_INIT = 0.25;  // 0.25 seconds
        
        if (time_since_last_kf >= KF_INTERVAL_BEFORE_IMU_INIT) {
            should_create_kf = true;
            LOG_DEBUG("Time-based keyframe: {:.3f}s since last KF", time_since_last_kf);
        }
        
        if (should_create_kf && result.success) {
            CreateKeyframe();
            m_last_keyframe_time = timestamp;
            
            // ORB-SLAM3 IMU init conditions: 10 keyframes AND 2 seconds
            constexpr int MIN_KF_FOR_IMU_INIT = 10;
            constexpr double MIN_TIME_FOR_IMU_INIT = 2.0;  // seconds
            
            double total_time = timestamp - m_first_keyframe_time;
            int num_keyframes = static_cast<int>(m_keyframes.size());
            
            LOG_DEBUG("IMU init check: {} KFs, {:.2f}s (need {} KFs, {:.1f}s)",
                     num_keyframes, total_time, MIN_KF_FOR_IMU_INIT, MIN_TIME_FOR_IMU_INIT);
            
            if (num_keyframes >= MIN_KF_FOR_IMU_INIT && total_time >= MIN_TIME_FOR_IMU_INIT) {
                LOG_INFO("IMU initialization conditions met! ({} KFs, {:.2f}s)",
                         num_keyframes, total_time);
                
                if (TryInitializeIMU()) {
                    // === Transition: VISUAL_ONLY -> VIO ===
                    m_tracking_state = TrackingState::VIO;
                    m_imu_init_frame_idx = m_current_frame->GetFrameId();  // Store frame index at IMU init
                    LOG_INFO("IMU Initialization SUCCESS! Transition to VIO state at frame {}.", m_imu_init_frame_idx);
                    
                    // Shrink window from init size to tracking size
                    ShrinkWindowAfterInit();
                }
            }
        }
        
        result.num_tracked = num_tracked;
        break;
    }
    
    case TrackingState::VIO: {
        // =================================================================
        // State: VIO - Full Visual-Inertial Odometry with IMU prediction
        // =================================================================
        int num_tracked = TrackFeatures();
        
        // Link MapPoints from previous frame
        LinkMapPointsFromPreviousFrame();
        
        // Count valid MapPoints
        int valid_mp_count = 0;
        const auto& features = m_current_frame->GetFeatures();
        for (size_t j = 0; j < features.size(); ++j) {
            auto mp = m_current_frame->GetMapPoint(static_cast<int>(j));
            if (mp && !mp->IsBad()) {
                valid_mp_count++;
            }
        }
        
        // IMU-aided pose prediction
        bool imu_predicted = PredictStateWithIMU();
        
        // Store IMU-predicted pose for comparison
        Eigen::Matrix4f imu_predicted_pose = Eigen::Matrix4f::Identity();
        Eigen::Vector3f imu_predicted_velocity = Eigen::Vector3f::Zero();
        
        if (!imu_predicted) {
            // Fallback to constant velocity model
            if (m_transform_from_last.isIdentity(1e-6)) {
                m_current_frame->SetTwb(m_previous_frame->GetTwb());
            } else {
                Eigen::Matrix4f predicted_pose = m_previous_frame->GetTwb() * m_transform_from_last;
                m_current_frame->SetTwb(predicted_pose);
            }
        } else {
            // Save IMU prediction for logging
            imu_predicted_pose = m_current_frame->GetTwb();
            imu_predicted_velocity = m_current_frame->GetVelocity();
        }
        
        // PnP pose estimation
        if (valid_mp_count >= 6) {
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(m_current_frame);
            
            if (pnp_result.success) {
                if (m_last_keyframe && !m_current_frame->IsKeyframe()) {
                    m_current_frame->SetReferenceKeyframe(m_last_keyframe);
                }
                
                m_current_pose = m_current_frame->GetTwb();
                m_transform_from_last = m_previous_frame->GetTwb().inverse() * m_current_pose;
                result.success = true;
                
                // Log IMU prediction vs PnP result
                if (imu_predicted) {
                    Eigen::Matrix4f pnp_pose = m_current_frame->GetTwb();
                    
                    // Position error
                    Eigen::Vector3f imu_pos = imu_predicted_pose.block<3,1>(0,3);
                    Eigen::Vector3f pnp_pos = pnp_pose.block<3,1>(0,3);
                    float pos_error = (imu_pos - pnp_pos).norm();
                    
                    // Rotation error (angle in degrees)
                    Eigen::Matrix3f imu_R = imu_predicted_pose.block<3,3>(0,0);
                    Eigen::Matrix3f pnp_R = pnp_pose.block<3,3>(0,0);
                    Eigen::Matrix3f R_err = imu_R.transpose() * pnp_R;
                    Eigen::AngleAxisf aa(R_err);
                    float rot_error_deg = aa.angle() * 180.0f / M_PI;
                    
                    LOG_INFO("Frame {}: pos_err={:.4f}m, rot_err={:.3f}deg",
                             m_current_frame->GetFrameId(), pos_error, rot_error_deg);
                }
            } else {
                LOG_WARN("Frame {}: PnP failed in VIO", m_current_frame->GetFrameId());
                result.success = false;
            }
        } else {
            LOG_WARN("Frame {}: Not enough MapPoints ({}) for PnP", 
                     m_current_frame->GetFrameId(), valid_mp_count);
            result.success = false;
        }
        
        // Detect new features if needed
        if (m_current_frame->GetFeatureCount() < 100) {
            DetectFeatures();
        }
        
        // Parallax-based keyframe creation (after IMU init)
        if (ShouldCreateKeyframe()) {
            CreateKeyframe();
        }
        
        result.num_tracked = num_tracked;
        break;
    }
    }  // end switch
    
    // Update state
    result.pose = m_current_pose;
    result.num_features = m_current_frame->GetFeatureCount();
    m_all_frames.push_back(m_current_frame);
    m_previous_frame = m_current_frame;
    
    return result;
}

void Estimator::ProcessIMU(const std::vector<IMUData>& imu_data) {
    if (imu_data.empty() || !m_current_frame || !m_previous_frame) {
        return;
    }
    
    // Get current bias estimates from previous frame
    Eigen::Vector3f accel_bias = m_previous_frame->GetAccelBias();
    Eigen::Vector3f gyro_bias = m_previous_frame->GetGyroBias();
    
    // Set bias in preintegrator
    m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
    
    // Get time range
    double start_time = imu_data.front().timestamp;
    double end_time = imu_data.back().timestamp;
    
    // Compute preintegration from last frame
    auto preint_from_last_frame = m_imu_preintegrator->Preintegrate(
        imu_data, start_time, end_time
    );
    
    // Store preintegration in current frame
    if (preint_from_last_frame) {
        m_current_frame->SetIMUPreintegrationFromLastFrame(preint_from_last_frame);
        
        // Copy bias from previous frame
        m_current_frame->SetAccelBias(accel_bias);
        m_current_frame->SetGyroBias(gyro_bias);
    }
    
    // Accumulate IMU data since last keyframe
    m_imu_since_last_keyframe.insert(
        m_imu_since_last_keyframe.end(),
        imu_data.begin(),
        imu_data.end()
    );
}

bool Estimator::PredictStateWithIMU() {
    if (m_tracking_state != TrackingState::VIO || !m_current_frame || !m_previous_frame) {
        return false;
    }
    
    // Get IMU preintegration from last frame
    auto preint = m_current_frame->GetIMUPreintegrationFromLastFrame();
    if (!preint || !preint->IsValid()) {
        return false;
    }
    
    // Get previous frame state
    Eigen::Matrix4f T_wb_prev = m_previous_frame->GetTwb();
    Eigen::Matrix3f R_wb_prev = T_wb_prev.block<3, 3>(0, 0);
    Eigen::Vector3f t_wb_prev = T_wb_prev.block<3, 1>(0, 3);
    Eigen::Vector3f v_prev = m_previous_frame->GetVelocity();
    
    // Get preintegration values
    float dt = static_cast<float>(preint->dt_total);
    Eigen::Matrix3f delta_R = preint->delta_R;
    Eigen::Vector3f delta_V = preint->delta_V;
    Eigen::Vector3f delta_P = preint->delta_P;
    
    // Predict current state using IMU preintegration
    // R_new = R_prev * delta_R
    Eigen::Matrix3f R_wb_curr = R_wb_prev * delta_R;
    
    // V_new = V_prev + g * dt + R_prev * delta_V
    Eigen::Vector3f v_curr = v_prev + m_gravity * dt + R_wb_prev * delta_V;
    
    // P_new = P_prev + V_prev * dt + 0.5 * g * dt^2 + R_prev * delta_P
    Eigen::Vector3f t_wb_curr = t_wb_prev + v_prev * dt + 0.5f * m_gravity * dt * dt + R_wb_prev * delta_P;
    
    // Set predicted state
    Eigen::Matrix4f T_wb_curr = Eigen::Matrix4f::Identity();
    T_wb_curr.block<3, 3>(0, 0) = R_wb_curr;
    T_wb_curr.block<3, 1>(0, 3) = t_wb_curr;
    
    m_current_frame->SetTwb(T_wb_curr);
    m_current_frame->SetVelocity(v_curr);
    
    return true;
}

bool Estimator::TryInitialize() {
    if (m_frame_window.size() < 2) {
        LOG_DEBUG("[Init] Window size {} < 2, skip", m_frame_window.size());
        return false;
    }
    
    // Get first and last frames in window
    auto first_frame = m_frame_window.front();
    auto last_frame = m_frame_window.back();
    
    // Compute parallax between first and last frames
    float parallax = m_initializer->ComputeParallax(first_frame, last_frame);
    
    LOG_DEBUG("[Init] Frame {}->{}: parallax={:.2f}px (min={:.2f})", 
             first_frame->GetFrameId(), last_frame->GetFrameId(), 
             parallax, m_min_parallax);
    
    // Check if parallax is sufficient
    if (parallax < m_min_parallax) {
        LOG_DEBUG("[Init] Insufficient parallax, waiting...");
        return false;
    }
    
    // Step 1: Select features with sufficient observations
    auto selected_features = m_initializer->SelectFeaturesForInit(m_frame_window);
    
    LOG_DEBUG("[Init] Selected {} features for initialization", selected_features.size());
    
    if (selected_features.empty()) {
        LOG_WARN("[Init] No features selected, skip");
        return false;
    }
    
    // Step 2: Try monocular initialization
    LOG_DEBUG("[Init] Attempting monocular initialization...");
    InitializationResult init_result;
    bool init_success = m_initializer->TryMonocularInitialization(m_frame_window, init_result);
    
    if (!init_success) {
        LOG_WARN("[Init] Monocular initialization failed");
        return false;
    }
    
    LOG_INFO("Monocular init: {} MapPoints created", init_result.initialized_mappoints.size());
    
    // Mark initialization MapPoints as marginalized (they define the scale)
    for (const auto& mp : init_result.initialized_mappoints) {
        if (mp) {
            mp->SetMarginalized(true);
        }
    }
    
    // Step 3: Store initialization results
    m_initialized_points = init_result.points3d;
    
    // Create pose matrices (Frame 1 at origin, Frame 2 with [R|t])
    m_init_poses.clear();
    m_init_poses.resize(2);
    
    // T_w1 = Identity (Frame 1 is at world origin)
    m_init_poses[0] = Eigen::Matrix4f::Identity();
    
    // T_w2 = [R | t]
    //        [0 | 1]
    m_init_poses[1] = Eigen::Matrix4f::Identity();
    m_init_poses[1].block<3, 3>(0, 0) = init_result.R;
    m_init_poses[1].block<3, 1>(0, 3) = init_result.t;
    
    // Return true to signal initialization is ready
    return true;
}

void Estimator::Reset() {
    m_current_frame = nullptr;
    m_previous_frame = nullptr;
    m_last_keyframe = nullptr;
    m_all_frames.clear();
    m_keyframes.clear();
    m_marginalized_poses.clear();
    m_frame_window.clear();
    m_frame_id_counter = 0;
    m_tracking_state = TrackingState::NOT_INITIALIZED;
    m_first_keyframe_time = -1.0;
    m_last_keyframe_time = -1.0;
    m_current_pose = Eigen::Matrix4f::Identity();
    m_transform_from_last = Eigen::Matrix4f::Identity();
    m_gravity = Eigen::Vector3f(0, 0, -9.81f);
    m_gyro_bias = Eigen::Vector3f::Zero();
    m_accel_bias = Eigen::Vector3f::Zero();
    m_scale = 1.0;
    m_imu_since_last_keyframe.clear();
}

std::shared_ptr<Frame> Estimator::CreateFrame(const cv::Mat& image, double timestamp) {
    const auto& config = ConfigUtils::GetInstance();
    
    // Frame constructor: timestamp (double in seconds), frame_id, image, width, height
    auto frame = std::make_shared<Frame>(
        timestamp,
        m_frame_id_counter++,
        image,
        config.camera_width,
        config.camera_height
    );
    
    // Set grid parameters
    frame->SetGridParameters(
        config.grid_cols,
        config.grid_rows,
        config.max_features_per_grid
    );
    
    // Set camera-to-body extrinsic transformation
    frame->SetTBC(config.T_BC);
    
    // Copy bias from previous frame (after IMU initialization)
    if (m_tracking_state == TrackingState::VIO && m_previous_frame) {
        frame->SetGyroBias(m_previous_frame->GetGyroBias());
        frame->SetAccelBias(m_previous_frame->GetAccelBias());
    }
    
    return frame;
}

int Estimator::TrackFeatures() {
    if (!m_previous_frame || !m_current_frame) {
        return 0;
    }
    
    // Track features using feature tracker
    m_feature_tracker->TrackFeatures(m_current_frame, m_previous_frame);
    
    // Get tracking stats
    int num_tracked, num_detected;
    m_feature_tracker->GetTrackingStats(num_tracked, num_detected);
    
    return num_tracked;
}

int Estimator::DetectFeatures() {
    if (!m_current_frame) {
        return 0;
    }
    
    // Detect features (pass nullptr as previous frame)
    m_feature_tracker->TrackFeatures(m_current_frame, nullptr);
    
    return m_current_frame->GetFeatureCount();
}

bool Estimator::ShouldCreateKeyframe() {
    if (!m_current_frame || !m_last_keyframe) {
        return false;
    }
    
    // Get parallax threshold from config
    const auto& config = ConfigUtils::GetInstance();
    float parallax_threshold = config.tracking_min_parallax_for_keyframe;
    
    // Compute parallax between last keyframe and current frame
    float parallax = ComputeParallax(m_last_keyframe, m_current_frame);
    
    // Create keyframe if parallax exceeds threshold
    if (parallax >= parallax_threshold) {
        LOG_DEBUG("Parallax {:.2f} >= {:.2f}, creating new keyframe", parallax, parallax_threshold);
        return true;
    }
    
    return false;
}

void Estimator::CreateKeyframe() {
    if (!m_current_frame) {
        return;
    }
    
    // Get previous keyframe before updating m_last_keyframe
    auto prev_keyframe = m_last_keyframe;
    
    // Compute and store preintegration from last keyframe BEFORE updating m_last_keyframe
    if (prev_keyframe && !m_imu_since_last_keyframe.empty()) {
        // Get bias from previous keyframe
        Eigen::Vector3f gyro_bias = prev_keyframe->GetGyroBias();
        Eigen::Vector3f accel_bias = prev_keyframe->GetAccelBias();
        m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
        
        // Compute preintegration from accumulated IMU data
        double start_time = m_imu_since_last_keyframe.front().timestamp;
        double end_time = m_imu_since_last_keyframe.back().timestamp;
        
        auto preint_from_last_kf = m_imu_preintegrator->Preintegrate(
            m_imu_since_last_keyframe, start_time, end_time
        );
        
        if (preint_from_last_kf) {
            m_current_frame->SetIMUPreintegrationFromLastKeyframe(preint_from_last_kf);
            LOG_DEBUG("  IMU: preintegration from kf {} to kf {}: dt={:.3f}s, {} measurements",
                     prev_keyframe->GetFrameId(), m_current_frame->GetFrameId(),
                     preint_from_last_kf->dt_total, m_imu_since_last_keyframe.size());
        }
    }
    
    // Clear accumulated IMU data for next keyframe
    m_imu_since_last_keyframe.clear();
    
    // Mark current frame as keyframe
    m_current_frame->SetKeyframe(true);
    m_keyframes.push_back(m_current_frame);
    m_last_keyframe = m_current_frame;
    
    // Add observations to MapPoints for this keyframe
    // (MapPoints linked from previous frames now get this keyframe as observer)
    const auto& features = m_current_frame->GetFeatures();
    int obs_added = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        auto mp = m_current_frame->GetMapPoint(static_cast<int>(i));
        if (mp && !mp->IsBad() && features[i] && features[i]->IsValid()) {
            // Check if this keyframe is already observing this MapPoint
            if (!mp->IsObservedByFrame(m_current_frame)) {
                mp->AddObservation(m_current_frame, static_cast<int>(i));
                obs_added++;
            }
        }
    }
    
    // Maintain keyframe window size (remove oldest keyframes from front)
    while (static_cast<int>(m_keyframes.size()) > m_window_size) {
        auto oldest_keyframe = m_keyframes.front();
        
        // Save pose before removing from window (for trajectory output)
        m_marginalized_poses.emplace_back(
            oldest_keyframe->GetTimestamp(),
            oldest_keyframe->GetTwb()
        );
        
        // For MapPoints whose reference keyframe is being removed:
        // Transfer reference to another keyframe in the window (keep MapPoint alive)
        const auto& old_kf_features = oldest_keyframe->GetFeatures();
        int transferred_count = 0;
        int deleted_count = 0;
        
        for (size_t i = 0; i < old_kf_features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                // Check if this keyframe is the reference (origin) for this MapPoint
                if (mp->IsReferenceKeyframe(oldest_keyframe)) {
                    // Find the oldest keyframe in window that observes this MapPoint
                    std::shared_ptr<Frame> new_ref_keyframe = nullptr;
                    
                    // Iterate through keyframes in window (oldest to newest, skipping the one being removed)
                    for (size_t kf_idx = 1; kf_idx < m_keyframes.size(); ++kf_idx) {
                        auto& kf = m_keyframes[kf_idx];
                        if (mp->IsObservedByFrame(kf)) {
                            new_ref_keyframe = kf;
                            break;  // Take the oldest one in window
                        }
                    }
                    
                    if (new_ref_keyframe) {
                        // Transfer reference to new keyframe
                        // Marginalize the MapPoint to preserve scale (its original reference defined the scale)
                        mp->SetReferenceKeyframe(new_ref_keyframe);
                        mp->SetMarginalized(true);  // Marginalize MapPoint when reference keyframe is removed
                        transferred_count++;
                    } else {
                        // No keyframe in window observes this MapPoint, mark as bad
                        mp->SetBad(true);
                        deleted_count++;
                    }
                }
            }
        }
        
        if (transferred_count > 0 || deleted_count > 0) {
            LOG_DEBUG("  Reference transfer: {} MapPoints moved, {} deleted (from removed kf {})", 
                     transferred_count, deleted_count, oldest_keyframe->GetFrameId());
        }
        
        // Clean up observations from MapPoints before removing the keyframe
        const auto& features = oldest_keyframe->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mp->RemoveObservation(oldest_keyframe);
                // Mark MapPoint as bad if no observations left
                if (mp->GetObservationCount() == 0) {
                    mp->SetBad(true);
                }
            }
        }
        
        m_keyframes.erase(m_keyframes.begin());  // Remove oldest (front)
    }
    
    // Triangulate new MapPoints between previous keyframe and current keyframe
    int new_points = 0;
    if (prev_keyframe) {
        new_points = TriangulateNewMapPoints(prev_keyframe, m_current_frame);
    }
    
    // Run BA if we have new MapPoints
    if (new_points > 0 && m_keyframes.size() >= 2) {
        Optimizer optimizer;
        optimizer.SetCamera(m_camera, m_boundary_margin);
        
        BAResult ba_result;
        if (m_tracking_state == TrackingState::VIO) {
            // Visual-Inertial BA with individual bias per frame
            ba_result = optimizer.RunLocalBAwithInertial(m_keyframes, m_gravity);
            
            // Get updated bias from last keyframe
            auto last_kf = m_keyframes.back();
            Eigen::Vector3f new_gyro_bias = last_kf->GetGyroBias();
            Eigen::Vector3f new_accel_bias = last_kf->GetAccelBias();
            
            // Update all preintegrations with new bias
            UpdatePreintegrationsWithNewBias(new_gyro_bias, new_accel_bias);
        } else {
            // Visual-only Local BA before IMU initialization
            ba_result = optimizer.RunLocalBA(m_keyframes);
        }
        
        // Update m_current_pose with BA-optimized pose
        m_current_pose = m_current_frame->GetTwb();
        
        // Reset m_transform_from_last to identity after BA
        // Next frame will start from the BA-optimized keyframe pose
        m_transform_from_last = Eigen::Matrix4f::Identity();
    }
    
    // Note: IMU initialization is now handled in ProcessFrame VISUAL_ONLY state
    // with ORB-SLAM3 style conditions (10 KF, 2 seconds)
}

void Estimator::LinkMapPointsFromPreviousFrame() {
    if (!m_current_frame || !m_previous_frame) {
        return;
    }
    
    const auto& curr_features = m_current_frame->GetFeatures();
    const auto& prev_features = m_previous_frame->GetFeatures();
    
    // Build map from feature_id to index in previous frame
    // Only include valid features (not outliers from PnP)
    std::unordered_map<int, size_t> prev_feature_map;
    int prev_with_mp = 0;
    for (size_t i = 0; i < prev_features.size(); ++i) {
        if (!prev_features[i]->IsValid()) continue;  // Skip outliers
        prev_feature_map[prev_features[i]->GetFeatureId()] = i;
        auto mp = m_previous_frame->GetMapPoint(static_cast<int>(i));
        if (mp && !mp->IsBad()) prev_with_mp++;
    }
    
    // Link MapPoints based on matching feature IDs
    int linked_count = 0;
    for (size_t i = 0; i < curr_features.size(); ++i) {
        int feat_id = curr_features[i]->GetFeatureId();
        
        auto it = prev_feature_map.find(feat_id);
        if (it != prev_feature_map.end()) {
            size_t prev_idx = it->second;
            auto mp = m_previous_frame->GetMapPoint(static_cast<int>(prev_idx));
            
            if (mp && !mp->IsBad()) {
                m_current_frame->SetMapPoint(static_cast<int>(i), mp);
                linked_count++;
            }
        }
    }
    
    LOG_DEBUG("LinkMapPoints: prev had {} MPs, linked {} to current frame", prev_with_mp, linked_count);
}

// DEPRECATED: This function was used for interpolation-based initialization.
// ORB-SLAM3 style uses real tracking instead. Kept for reference but not called.
void Estimator::ProcessIntermediateFrames() {
    if (m_frame_window.size() < 3) {
        return;  // No intermediate frames
    }
    
    auto first_kf = m_frame_window.front();
    auto last_kf = m_frame_window.back();
    
    Eigen::Matrix4f T_first = first_kf->GetTwb();
    Eigen::Matrix4f T_last = last_kf->GetTwb();
    
    // Extract rotation and translation
    Eigen::Matrix3f R_first = T_first.block<3, 3>(0, 0);
    Eigen::Vector3f t_first = T_first.block<3, 1>(0, 3);
    Eigen::Matrix3f R_last = T_last.block<3, 3>(0, 0);
    Eigen::Vector3f t_last = T_last.block<3, 1>(0, 3);
    
    // Compute relative rotation using Rodrigues
    Eigen::Matrix3f R_rel = R_first.transpose() * R_last;
    Eigen::AngleAxisf aa(R_rel);
    Eigen::Vector3f axis = aa.axis();
    float angle = aa.angle();
    
    int n_frames = static_cast<int>(m_frame_window.size());
    
    LOG_DEBUG("Processing {} intermediate frames to create keyframes...", n_frames - 2);
    
    // Build global feature ID to MapPoint map from first_kf and last_kf (the two frames with triangulated MapPoints)
    std::unordered_map<int, std::shared_ptr<MapPoint>> feature_to_mappoint;
    
    // Collect from first keyframe
    const auto& first_features = first_kf->GetFeatures();
    for (size_t j = 0; j < first_features.size(); ++j) {
        auto mp = first_kf->GetMapPoint(static_cast<int>(j));
        if (mp && !mp->IsBad()) {
            feature_to_mappoint[first_features[j]->GetFeatureId()] = mp;
        }
    }
    
    // Collect from last keyframe (may add new MapPoints or override)
    const auto& last_features = last_kf->GetFeatures();
    for (size_t j = 0; j < last_features.size(); ++j) {
        auto mp = last_kf->GetMapPoint(static_cast<int>(j));
        if (mp && !mp->IsBad()) {
            int feat_id = last_features[j]->GetFeatureId();
            if (feature_to_mappoint.find(feat_id) == feature_to_mappoint.end()) {
                feature_to_mappoint[feat_id] = mp;
            }
        }
    }
    
    LOG_DEBUG("Found {} unique MapPoints from first/last keyframes for observation propagation", 
             feature_to_mappoint.size());
    
    // Process intermediate frames (skip first and last which already have poses)
    for (int i = 1; i < n_frames - 1; ++i) {
        auto& frame = m_frame_window[i];
        
        // Interpolate pose (linear interpolation for translation, slerp for rotation)
        float alpha = static_cast<float>(i) / static_cast<float>(n_frames - 1);
        
        // Interpolate rotation using axis-angle
        float interp_angle = alpha * angle;
        Eigen::Matrix3f R_interp = R_first * Eigen::AngleAxisf(interp_angle, axis).toRotationMatrix();
        
        // Interpolate translation
        Eigen::Vector3f t_interp = (1.0f - alpha) * t_first + alpha * t_last;
        
        // Set interpolated pose
        Eigen::Matrix4f T_interp = Eigen::Matrix4f::Identity();
        T_interp.block<3, 3>(0, 0) = R_interp;
        T_interp.block<3, 1>(0, 3) = t_interp;
        frame->SetTwb(T_interp);
        
        // Link MapPoints from global feature->mappoint map (matches feature IDs across all triangulated points)
        const auto& curr_features = frame->GetFeatures();
        
        int linked = 0;
        for (size_t j = 0; j < curr_features.size(); ++j) {
            int feat_id = curr_features[j]->GetFeatureId();
            auto it = feature_to_mappoint.find(feat_id);
            if (it != feature_to_mappoint.end()) {
                auto mp = it->second;
                if (mp && !mp->IsBad()) {
                    frame->SetMapPoint(static_cast<int>(j), mp);
                    mp->AddObservation(frame, static_cast<int>(j));
                    linked++;
                }
            }
        }
        
        LOG_INFO("  Frame {} (idx {}): linked {} MapPoints", frame->GetFrameId(), i, linked);
        
        // Run PnP to refine pose
        if (linked >= 6) {
            Optimizer optimizer;
            optimizer.SetCamera(m_camera, m_boundary_margin);
            PnPResult pnp_result = optimizer.SolvePnP(frame);
            LOG_INFO("  Frame {} PnP: {} inliers, {} outliers, reproj={:.2f}px",
                     frame->GetFrameId(), pnp_result.num_inliers, pnp_result.num_outliers,
                     pnp_result.final_cost);
        } else {
            LOG_WARN("  Frame {} (idx {}): only {} linked points, skipping PnP", 
                    frame->GetFrameId(), i, linked);
        }
    }
    
    // Run BA on all frames in window (first frame fixed, last frame NOT fixed)
    LOG_INFO("Running BA on {} frames with all MapPoints...", n_frames);
    Optimizer optimizer;
    optimizer.SetCamera(m_camera, m_boundary_margin);
    BAResult ba_result = optimizer.RunBA(m_frame_window, true, false);
    LOG_INFO("Init window BA: {} inliers, {} outliers, cost {:.2f} -> {:.2f}",
             ba_result.num_inliers, ba_result.num_outliers,
             ba_result.initial_cost, ba_result.final_cost);
    
    // Mark all frames as keyframes
    for (int i = 0; i < n_frames; ++i) {
        auto& frame = m_frame_window[i];
        frame->SetKeyframe(true);
    }
    
    // Use preintegration from last frame (already computed per-frame during tracking)
    // In init phase, each frame has preintegration from previous frame
    // For keyframes: just copy frame-to-frame preintegration as keyframe-to-keyframe
    LOG_INFO("Setting up IMU preintegration for {} keyframe pairs...", n_frames - 1);
    int valid_preint_count = 0;
    for (int i = 1; i < n_frames; ++i) {
        auto& curr_kf = m_frame_window[i];
        
        // Get preintegration from last frame (which is also the previous keyframe in init phase)
        auto preint = curr_kf->GetIMUPreintegrationFromLastFrame();
        if (preint) {
            curr_kf->SetIMUPreintegrationFromLastKeyframe(preint);
            valid_preint_count++;
            LOG_INFO("  IMU preint[{}]: kf{} -> kf{}: dt={:.3f}s, dV=({:.3f},{:.3f},{:.3f})",
                     i-1, m_frame_window[i-1]->GetFrameId(), curr_kf->GetFrameId(),
                     preint->dt_total,
                     preint->delta_V.x(), preint->delta_V.y(), preint->delta_V.z());
        } else {
            LOG_WARN("  No preintegration for kf{} -> kf{}",
                    m_frame_window[i-1]->GetFrameId(), curr_kf->GetFrameId());
        }
    }
    LOG_INFO("IMU preintegration: {}/{} pairs ready", valid_preint_count, n_frames - 1);
    
    // Clear accumulated IMU data (not needed anymore for init)
    m_imu_since_last_keyframe.clear();
    
    // Add all keyframes to the list (first and last already added in caller)
    // Clear existing and re-add all in order
    m_keyframes.clear();
    m_marginalized_poses.clear();  // Clear old marginalized poses on IMU init
    for (int i = 0; i < n_frames; ++i) {
        m_keyframes.push_back(m_frame_window[i]);
    }
    m_last_keyframe = m_frame_window.back();
    
    // Log MapPoint count for each keyframe
    for (int i = 0; i < n_frames; ++i) {
        auto& kf = m_frame_window[i];
        int mp_count = 0;
        size_t n_features = kf->GetFeatureCount();
        for (size_t j = 0; j < n_features; ++j) {
            auto mp = kf->GetMapPoint(j);
            if (mp && !mp->IsBad()) {
                mp_count++;
            }
        }
        LOG_INFO("KF[{}] frame_id={}: {} MapPoints (features={})", i, kf->GetFrameId(), mp_count, n_features);
    }
    
    LOG_INFO("Created {} keyframes from initialization window", n_frames);
}

float Estimator::ComputeParallax(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2
) const {
    if (!frame1 || !frame2) {
        return 0.0f;
    }
    
    const auto& features1 = frame1->GetFeatures();
    const auto& features2 = frame2->GetFeatures();
    
    if (features1.empty() || features2.empty()) {
        return 0.0f;
    }
    
    // Build map from feature_id to pixel coord in frame1
    std::unordered_map<int, cv::Point2f> feat1_map;
    for (const auto& feat : features1) {
        feat1_map[feat->GetFeatureId()] = feat->GetPixelCoord();
    }
    
    // Find correspondences and compute parallax
    std::vector<float> parallaxes;
    parallaxes.reserve(features2.size());
    
    for (const auto& feat2 : features2) {
        auto it = feat1_map.find(feat2->GetFeatureId());
        if (it != feat1_map.end()) {
            const cv::Point2f& pt1 = it->second;
            cv::Point2f pt2 = feat2->GetPixelCoord();
            
            float dx = pt2.x - pt1.x;
            float dy = pt2.y - pt1.y;
            float parallax = std::sqrt(dx * dx + dy * dy);
            
            parallaxes.push_back(parallax);
        }
    }
    
    if (parallaxes.empty()) {
        return 0.0f;
    }
    
    // Return median parallax (more robust than mean)
    std::sort(parallaxes.begin(), parallaxes.end());
    size_t mid = parallaxes.size() / 2;
    
    if (parallaxes.size() % 2 == 0) {
        return (parallaxes[mid - 1] + parallaxes[mid]) / 2.0f;
    } else {
        return parallaxes[mid];
    }
}

bool Estimator::IsKeyframeInWindow(int frame_id) const {
    for (const auto& kf : m_keyframes) {
        if (kf->GetFrameId() == frame_id) return true;
    }
    return false;
}

bool Estimator::TriangulateSinglePoint(
    const Eigen::Vector3f& bearing1,
    const Eigen::Vector3f& bearing2,
    const Eigen::Matrix4f& T1w,
    const Eigen::Matrix4f& T2w,
    Eigen::Vector3f& point3d
) const {
    // Triangulation using SVD
    
    // Build 4x4 matrix A for SVD triangulation
    // A.row(0) = bearing_1(0) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(0);
    // A.row(1) = bearing_1(1) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(1);
    // A.row(2) = bearing_2(0) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(0);
    // A.row(3) = bearing_2(1) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(1);
    
    Eigen::Matrix4f A;
    A.row(0) = bearing1(0) * T1w.row(2) - bearing1(2) * T1w.row(0);
    A.row(1) = bearing1(1) * T1w.row(2) - bearing1(2) * T1w.row(1);
    A.row(2) = bearing2(0) * T2w.row(2) - bearing2(2) * T2w.row(0);
    A.row(3) = bearing2(1) * T2w.row(2) - bearing2(2) * T2w.row(1);
    
    // SVD decomposition
    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4f singular_vector = svd.matrixV().col(3);
    
    // Check for valid homogeneous coordinate
    if (std::abs(singular_vector(3)) < 1e-10f) {
        return false;
    }
    
    // Convert to 3D point
    point3d = singular_vector.head<3>() / singular_vector(3);
    
    // Equirectangular cameras skip depth check
    // because they can see in all directions (no "behind camera" concept)
    
    // Only check for valid (finite) coordinates and reasonable range
    if (!std::isfinite(point3d.x()) || !std::isfinite(point3d.y()) || !std::isfinite(point3d.z())) {
        return false;
    }
    
    
    return true;
}

int Estimator::TriangulateNewMapPoints(
    const std::shared_ptr<Frame>& kf1,
    const std::shared_ptr<Frame>& kf2
) {
    if (!kf1 || !kf2) {
        return 0;
    }
    
    const auto& features1 = kf1->GetFeatures();
    const auto& features2 = kf2->GetFeatures();
    
    LOG_DEBUG("Triangulation: kf1={} ({} feats), kf2={} ({} feats)",
             kf1->GetFrameId(), features1.size(), kf2->GetFrameId(), features2.size());
    
    // Build map: feature_id -> (index in kf1, feature pointer)
    // Only include valid features (not outliers)
    std::unordered_map<int, std::pair<int, std::shared_ptr<Feature>>> kf1_map;
    for (size_t i = 0; i < features1.size(); ++i) {
        if (!features1[i]->IsValid()) continue;  // Skip outliers
        int feat_id = features1[i]->GetFeatureId();
        kf1_map[feat_id] = {static_cast<int>(i), features1[i]};
    }
    
    // Get poses (world to camera)
    // GetTwc() returns camera to world, so we need to invert it
    Eigen::Matrix4f T1w = kf1->GetTwc().inverse();  // World to camera1
    Eigen::Matrix4f T2w = kf2->GetTwc().inverse();  // World to camera2
    
    // Debug: print baseline
    Eigen::Matrix3f R1 = T1w.block<3, 3>(0, 0);
    Eigen::Vector3f t1 = T1w.block<3, 1>(0, 3);
    Eigen::Vector3f C1 = -R1.transpose() * t1;
    
    Eigen::Matrix3f R2 = T2w.block<3, 3>(0, 0);
    Eigen::Vector3f t2 = T2w.block<3, 1>(0, 3);
    Eigen::Vector3f C2 = -R2.transpose() * t2;
    
    float baseline = (C1 - C2).norm();
    LOG_DEBUG("  Baseline: {:.6f}, C1=({:.4f},{:.4f},{:.4f}), C2=({:.4f},{:.4f},{:.4f})",
             baseline, C1.x(), C1.y(), C1.z(), C2.x(), C2.y(), C2.z());
    
    int triangulated_count = 0;
    int matched_count = 0;
    int already_has_mp = 0;
    int depth_failed = 0;
    int reproj_failed = 0;
    int parallax_failed = 0;
    
    
    for (size_t i2 = 0; i2 < features2.size(); ++i2) {
        // Skip invalid features (outliers)
        if (!features2[i2]->IsValid()) continue;
        
        int feat_id = features2[i2]->GetFeatureId();
        
        // Check if this feature exists in kf1
        auto it = kf1_map.find(feat_id);
        if (it == kf1_map.end()) {
            continue;
        }
        
        matched_count++;
        int i1 = it->second.first;
        
        // Check if already has MapPoint
        auto existing_mp = kf2->GetMapPoint(static_cast<int>(i2));
        if (existing_mp && !existing_mp->IsBad()) {
            already_has_mp++;
            continue;  // Already has valid MapPoint
        }
        
        // Get bearing vectors
        Eigen::Vector3f bearing1 = features1[i1]->GetBearing();
        Eigen::Vector3f bearing2 = features2[i2]->GetBearing();
        
        // Check parallax (angle between bearing vectors in world frame)
        // Transform bearings to world frame
        Eigen::Matrix3f R1_wc = kf1->GetTwc().block<3, 3>(0, 0);
        Eigen::Matrix3f R2_wc = kf2->GetTwc().block<3, 3>(0, 0);
        Eigen::Vector3f bearing1_world = R1_wc * bearing1;
        Eigen::Vector3f bearing2_world = R2_wc * bearing2;
        
        float cos_parallax = bearing1_world.dot(bearing2_world);
        cos_parallax = std::max(-1.0f, std::min(1.0f, cos_parallax));
        float parallax_deg = std::acos(cos_parallax) * 180.0f / M_PI;
        
        // Skip if parallax is too small (< 0.1 degree)
        const float min_parallax_deg = 0.1f;
        if (parallax_deg < min_parallax_deg) {
            parallax_failed++;
            continue;
        }
        
        // Triangulate
        Eigen::Vector3f point3d;
        if (!TriangulateSinglePoint(bearing1, bearing2, T1w, T2w, point3d)) {
            depth_failed++;
            continue;
        }
        
        // Verify reprojection error before accepting the point
        // Transform point to both camera frames and check error
        Eigen::Vector4f point3d_h(point3d.x(), point3d.y(), point3d.z(), 1.0f);
        
        // Camera 1 reprojection
        Eigen::Vector4f pc1_h = T1w * point3d_h;
        Eigen::Vector3f pc1 = pc1_h.head<3>();
        Eigen::Vector3f reproj_bearing1 = pc1.normalized();
        float dot1 = bearing1.dot(reproj_bearing1);
        float angle_error1 = std::acos(std::min(1.0f, std::abs(dot1)));
        float pixel_error1 = angle_error1 * kf1->GetWidth() / (2.0f * M_PI);
        
        // Camera 2 reprojection
        Eigen::Vector4f pc2_h = T2w * point3d_h;
        Eigen::Vector3f pc2 = pc2_h.head<3>();
        Eigen::Vector3f reproj_bearing2 = pc2.normalized();
        float dot2 = bearing2.dot(reproj_bearing2);
        float angle_error2 = std::acos(std::min(1.0f, std::abs(dot2)));
        float pixel_error2 = angle_error2 * kf2->GetWidth() / (2.0f * M_PI);
        
        // Reject if reprojection error is too large
        // Chi-squared threshold: chi_sq_2D * sigma_sq = 5.99 * 1.0 ≈ 2.5 px
        // // For equirectangular with more noise, use relaxed threshold
        // const float max_reproj_error = 5.0f;
        // if (pixel_error1 > max_reproj_error || pixel_error2 > max_reproj_error) {
        //     reproj_failed++;
        //     continue;
        // }
        
        // Create new MapPoint
        auto mp = std::make_shared<MapPoint>(point3d);
        mp->SetTriangulated(true);
        
        // Set the earlier keyframe (kf1) as the reference keyframe for scale consistency
        mp->SetReferenceKeyframe(kf1);
        
        // Add observations to MapPoint
        mp->AddObservation(kf1, i1);
        mp->AddObservation(kf2, static_cast<int>(i2));
        
        // Register MapPoint to keyframes
        kf1->SetMapPoint(i1, mp);
        kf2->SetMapPoint(static_cast<int>(i2), mp);
        
        // Also add observations from intermediate frames in the sliding window
        // The feature in kf2 may have been tracked through multiple frames
        const auto& feat_observations = features2[i2]->GetObservations();
        for (const auto& obs : feat_observations) {
            auto obs_frame = obs.frame;
            if (!obs_frame) continue;
            
            // Skip if it's kf1 or kf2 (already added)
            if (obs_frame->GetFrameId() == kf1->GetFrameId() ||
                obs_frame->GetFrameId() == kf2->GetFrameId()) {
                continue;
            }
            
            // Only add if the frame is a keyframe in the current window
            if (!obs_frame->IsKeyframe()) continue;
            if (!IsKeyframeInWindow(obs_frame->GetFrameId())) continue;
            
            int obs_feat_idx = obs.feature_index;
            
            // Verify the observation is valid (check reprojection error)
            const auto& obs_features = obs_frame->GetFeatures();
            if (obs_feat_idx < 0 || obs_feat_idx >= static_cast<int>(obs_features.size())) {
                spdlog::error("Invalid feature index {} in frame {}", obs_feat_idx, obs_frame->GetFrameId());
                continue;
            }
            
            Eigen::Vector3f obs_bearing = obs_features[obs_feat_idx]->GetBearing();
            Eigen::Matrix4f T_obs_w = obs_frame->GetTwc().inverse();
            Eigen::Vector4f pt_h(point3d.x(), point3d.y(), point3d.z(), 1.0f);
            Eigen::Vector3f pt_cam = (T_obs_w * pt_h).head<3>();
            Eigen::Vector3f reproj_bearing = pt_cam.normalized();
            
            float dot = obs_bearing.dot(reproj_bearing);
            float angle_error = std::acos(std::min(1.0f, std::abs(dot)));
            float pixel_error = angle_error * obs_frame->GetWidth() / (2.0f * M_PI);
            
            // Only add if reprojection error is acceptable
            // if (pixel_error <= max_reproj_error) 
            {
                mp->AddObservation(obs_frame, obs_feat_idx);
                obs_frame->SetMapPoint(obs_feat_idx, mp);
            }
        }
        
        triangulated_count++;
    }
    
    LOG_DEBUG("  Matched: {}, already_has_mp: {}, parallax_fail: {}, depth_fail: {}, success: {}",
             matched_count, already_has_mp, parallax_failed, depth_failed, triangulated_count);
    
    return triangulated_count;
}

bool Estimator::TryInitializeIMU() {
    // Already initialized?
    if (m_tracking_state == TrackingState::VIO) {
        return true;
    }
    
    // Need at least 3 keyframes for IMU initialization
    if (m_keyframes.size() < 3) {
        LOG_INFO("[IMU_INIT] Waiting for more keyframes: {}/3", m_keyframes.size());
        return false;
    }
    
    // Check if all keyframes have preintegration (except first)
    bool all_have_preint = true;
    for (size_t i = 1; i < m_keyframes.size(); ++i) {
        if (!m_keyframes[i]->HasIMUPreintegrationFromLastKeyframe()) {
            LOG_INFO("[IMU_INIT] Keyframe {} missing preintegration", i);
            all_have_preint = false;
            break;
        }
    }
    
    if (!all_have_preint) {
        LOG_INFO("[IMU_INIT] Not all keyframes have preintegration");
        return false;
    }
    
    LOG_INFO("[IMU_INIT] Starting IMU initialization with {} keyframes", m_keyframes.size());
    
    // Run 2-stage IMU optimization (gravity + scale + velocities + biases)
    Optimizer optimizer;
    auto result = optimizer.OptimizeIMUInit(m_keyframes);
    
    if (!result.success) {
        LOG_WARN("[IMU_INIT] IMU initialization failed");
        return false;
    }
    
    // Store results
    m_gravity = result.gravity;
    m_scale = result.scale;
    m_gyro_bias = result.gyro_bias;
    m_accel_bias = result.accel_bias;
    
    // Apply scaled poses from optimizer (ORB-SLAM3 style: scale applied internally)
    for (size_t i = 0; i < m_keyframes.size() && i < result.scaled_poses.size(); ++i) {
        m_keyframes[i]->SetTwb(result.scaled_poses[i]);
    }
    
    // Update velocities and biases in keyframes (before transformation)
    for (size_t i = 0; i < m_keyframes.size() && i < result.velocities.size(); ++i) {
        m_keyframes[i]->SetVelocity(result.velocities[i]);
        m_keyframes[i]->SetGyroBias(m_gyro_bias);
        m_keyframes[i]->SetAccelBias(m_accel_bias);
    }
    
    // Update biases in preintegrator
    m_imu_preintegrator->SetBias(m_gyro_bias, m_accel_bias);
    
    // Update current frame bias if exists
    if (m_current_frame) {
        m_current_frame->SetGyroBias(m_gyro_bias);
        m_current_frame->SetAccelBias(m_accel_bias);
    }
    
    LOG_INFO("[IMU_INIT] IMU initialization SUCCESS!");
    LOG_INFO("  Gravity (before transform): [{:.4f}, {:.4f}, {:.4f}] (norm={:.4f})",
             m_gravity.x(), m_gravity.y(), m_gravity.z(), m_gravity.norm());
    LOG_INFO("  Scale: {:.4f}", m_scale);
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}]",
             m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
    LOG_INFO("  Accel bias: [{:.6f}, {:.6f}, {:.6f}]",
             m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
    
    // Apply gravity alignment transformation to world coordinate system
    // This transforms all poses, MapPoints, and velocities so gravity points in -Z
    // Scale is already applied in Optimizer (both poses and MapPoints), so pass 1.0 here
    ApplyGravityAlignmentTransform(result.Rwg, 1.0f);
    
    // Update gravity to point in -Z direction (after transformation)
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    
    // Note: m_tracking_state transition to VIO is done in ProcessFrame after TryInitializeIMU returns true
    
    return true;
}

bool Estimator::TryInitializeIMUWithInterpolation() {
    // =========================================================================
    // Interpolation-Based IMU Initialization
    // Uses intermediate frames between first two keyframes with interpolated poses
    // =========================================================================
    
    if (m_tracking_state == TrackingState::VIO) {
        return true;  // Already initialized
    }
    
    // Need at least 2 keyframes (first and last from frame_window)
    if (m_keyframes.size() < 2) {
        LOG_INFO("[IMU_INIT_INTERP] Need at least 2 keyframes, got {}", m_keyframes.size());
        return false;
    }
    
    // Check if we have intermediate frames in frame_window
    if (m_frame_window.size() < 3) {
        LOG_INFO("[IMU_INIT_INTERP] Need intermediate frames in window, got {}", m_frame_window.size());
        return false;
    }
    
    // Get first and last keyframes
    auto kf1 = m_frame_window.front();  // First keyframe
    auto kf2 = m_frame_window.back();   // Last keyframe
    
    Eigen::Matrix4f T1 = kf1->GetTwb();
    Eigen::Matrix4f T2 = kf2->GetTwb();
    
    // Extract rotation and translation
    Eigen::Matrix3f R1 = T1.block<3,3>(0,0);
    Eigen::Vector3f t1 = T1.block<3,1>(0,3);
    Eigen::Matrix3f R2 = T2.block<3,3>(0,0);
    Eigen::Vector3f t2 = T2.block<3,1>(0,3);
    
    // Relative rotation for interpolation
    Eigen::Matrix3f R_rel = R1.transpose() * R2;
    Eigen::AngleAxisf aa(R_rel);
    Eigen::Vector3f axis = aa.axis();
    float angle = aa.angle();
    
    int n_frames = static_cast<int>(m_frame_window.size());
    
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT_INTERP] Interpolation-Based IMU Initialization");
    LOG_INFO("  Total frames: {}, KF1 id: {}, KF2 id: {}",
             n_frames, kf1->GetFrameId(), kf2->GetFrameId());
    LOG_INFO("========================================================");
    
    // =========================================================================
    // STEP 1: Interpolate poses for intermediate frames
    // =========================================================================
    
    double total_dt = kf2->GetTimestamp() - kf1->GetTimestamp();
    if (total_dt < 0.01) {
        LOG_WARN("[IMU_INIT_INTERP] Time span too short: {:.3f}s", total_dt);
        return false;
    }
    
    std::vector<Eigen::Matrix4f> interpolated_poses(n_frames);
    std::vector<double> frame_times(n_frames);
    
    interpolated_poses[0] = T1;
    frame_times[0] = kf1->GetTimestamp();
    
    for (int i = 1; i < n_frames - 1; ++i) {
        auto& frame = m_frame_window[i];
        double t_frame = frame->GetTimestamp();
        float alpha = static_cast<float>((t_frame - kf1->GetTimestamp()) / total_dt);
        alpha = std::max(0.0f, std::min(1.0f, alpha));
        
        // SLERP for rotation
        float interp_angle = alpha * angle;
        Eigen::Matrix3f R_interp = R1 * Eigen::AngleAxisf(interp_angle, axis).toRotationMatrix();
        
        // LERP for translation
        Eigen::Vector3f t_interp = (1.0f - alpha) * t1 + alpha * t2;
        
        Eigen::Matrix4f T_interp = Eigen::Matrix4f::Identity();
        T_interp.block<3,3>(0,0) = R_interp;
        T_interp.block<3,1>(0,3) = t_interp;
        
        interpolated_poses[i] = T_interp;
        frame_times[i] = t_frame;
        frame->SetTwb(T_interp);
    }
    
    interpolated_poses[n_frames - 1] = T2;
    frame_times[n_frames - 1] = kf2->GetTimestamp();
    
    LOG_INFO("  Interpolated {} intermediate frame poses", n_frames - 2);
    
    // =========================================================================
    // STEP 2: Compute frame-to-frame preintegrations
    // =========================================================================
    
    std::vector<std::shared_ptr<IMUPreintegration>> preints;
    std::vector<double> dts;
    int valid_preint_count = 0;
    
    for (int i = 0; i < n_frames - 1; ++i) {
        auto& frame_i = m_frame_window[i];
        auto& frame_j = m_frame_window[i + 1];
        
        double t_start = frame_times[i];
        double t_end = frame_times[i + 1];
        double dt = t_end - t_start;
        
        // Look for IMU data for this frame
        auto it = m_imu_data_per_frame.find(frame_j->GetFrameId());
        if (it == m_imu_data_per_frame.end() || it->second.empty()) {
            // Try getting from keyframe preintegration
            auto kf_preint = frame_j->GetIMUPreintegrationFromLastKeyframe();
            if (kf_preint && kf_preint->IsValid()) {
                preints.push_back(kf_preint);
                dts.push_back(kf_preint->dt_total);
                valid_preint_count++;
                continue;
            }
            
            LOG_DEBUG("  Frame {} has no IMU data", frame_j->GetFrameId());
            preints.push_back(nullptr);
            dts.push_back(dt);
            continue;
        }
        
        // Preintegrate for this frame pair
        Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
        Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
        m_imu_preintegrator->SetBias(gyro_bias, accel_bias);
        
        auto preint = m_imu_preintegrator->Preintegrate(it->second, t_start, t_end);
        if (preint && preint->IsValid()) {
            preints.push_back(preint);
            dts.push_back(preint->dt_total);
            valid_preint_count++;
        } else {
            preints.push_back(nullptr);
            dts.push_back(dt);
        }
    }
    
    LOG_INFO("  Valid preintegrations: {} / {}", valid_preint_count, n_frames - 1);
    
    if (valid_preint_count < 2) {
        LOG_WARN("[IMU_INIT_INTERP] Not enough preintegrations for initialization");
        return false;
    }
    
    // =========================================================================
    // STEP 3: Gyro Bias Estimation (Rotation-only)
    // =========================================================================
    
    Eigen::Matrix3d A_gyro = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_gyro = Eigen::Vector3d::Zero();
    int gyro_pairs = 0;
    
    for (int i = 0; i < n_frames - 1; ++i) {
        if (!preints[i] || !preints[i]->IsValid()) continue;
        
        // Visual rotation: R_ij = R_i^T * R_j
        Eigen::Matrix3d R_i = interpolated_poses[i].block<3,3>(0,0).cast<double>();
        Eigen::Matrix3d R_j = interpolated_poses[i+1].block<3,3>(0,0).cast<double>();
        Eigen::Matrix3d R_ij_visual = R_i.transpose() * R_j;
        
        // IMU rotation
        Eigen::Matrix3d delta_R_imu = preints[i]->delta_R.cast<double>();
        
        // Rotation error
        Eigen::Matrix3d R_err = delta_R_imu.transpose() * R_ij_visual;
        Eigen::AngleAxisd aa_err(R_err);
        Eigen::Vector3d rotation_error = aa_err.angle() * aa_err.axis();
        
        // Jacobian
        Eigen::Matrix3d J_Rg = preints[i]->J_Rg.cast<double>();
        
        A_gyro += J_Rg.transpose() * J_Rg;
        b_gyro += J_Rg.transpose() * rotation_error;
        gyro_pairs++;
    }
    
    Eigen::Vector3d gyro_bias_est = Eigen::Vector3d::Zero();
    if (gyro_pairs > 0 && A_gyro.determinant() > 1e-10) {
        gyro_bias_est = A_gyro.ldlt().solve(b_gyro);
    }
    
    m_gyro_bias = gyro_bias_est.cast<float>();
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}] rad/s ({} pairs)",
             m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z(), gyro_pairs);
    
    // =========================================================================
    // STEP 4: Gravity Direction Estimation (ORB-SLAM3 style)
    // Use delta_V from preintegration: delta_V = R_i^T * (v_j - v_i - g * dt)
    // Rearranging: R_i * delta_V = v_j - v_i - g * dt
    // Sum over all frames: gravity direction is -sum(R_i * delta_V) / sum(dt)
    // =========================================================================
    
    Eigen::Vector3d dirG = Eigen::Vector3d::Zero();
    double total_dt_gravity = 0.0;
    int gravity_samples = 0;
    
    for (int i = 0; i < n_frames - 1; ++i) {
        if (!preints[i] || !preints[i]->IsValid()) continue;
        
        double dt = preints[i]->dt_total;
        if (dt < 0.001) continue;
        
        Eigen::Matrix3d R_i = interpolated_poses[i].block<3,3>(0,0).cast<double>();
        
        // Corrected delta_V with gyro bias
        Eigen::Vector3d delta_V = preints[i]->delta_V.cast<double>();
        Eigen::Matrix3d J_Vg = preints[i]->J_Vg.cast<double>();
        delta_V += J_Vg * gyro_bias_est;
        
        // ORB-SLAM3: dirG = -sum(R_i * delta_V)
        // This accumulates the acceleration direction
        dirG -= R_i * delta_V;
        total_dt += dt;
        gravity_samples++;
    }
    
    if (gravity_samples == 0 || total_dt < 0.01) {
        LOG_WARN("[IMU_INIT_INTERP] No valid samples for gravity estimation");
        return false;
    }
    
    // ORB-SLAM3 style: gravity direction from accumulated delta_V
    // dirG points in the gravity direction (scaled by time)
    Eigen::Vector3d gravity_dir = dirG / total_dt;  // Average acceleration
    double gravity_norm = gravity_dir.norm();
    
    LOG_INFO("  Raw gravity: [{:.4f}, {:.4f}, {:.4f}] (norm={:.4f})",
             gravity_dir.x(), gravity_dir.y(), gravity_dir.z(), gravity_norm);
    
    // Normalize to 9.81
    if (gravity_norm > 0.1) {
        gravity_dir = gravity_dir.normalized() * 9.81;
    } else {
        gravity_dir = Eigen::Vector3d(0, 0, -9.81);  // Default
    }
    
    m_gravity = gravity_dir.cast<float>();
    
    // =========================================================================
    // STEP 6: Compute gravity alignment rotation (Rwg)
    // =========================================================================
    
    Eigen::Vector3d gz(0, 0, -1);  // Target gravity direction
    Eigen::Vector3d gn = gravity_dir.normalized();
    
    Eigen::Matrix3d Rwg = Eigen::Matrix3d::Identity();
    double dot = gz.dot(gn);
    
    if (std::abs(dot + 1.0) < 1e-6) {
        // Anti-parallel: rotate 180° around any perpendicular axis
        Rwg = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
    } else if (std::abs(dot - 1.0) > 1e-6) {
        // General case: use Rodrigues
        Eigen::Vector3d axis = gn.cross(gz);
        double axis_norm = axis.norm();
        if (axis_norm > 1e-6) {
            axis.normalize();
            double angle = std::acos(std::max(-1.0, std::min(1.0, dot)));
            Rwg = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
    }
    
    // =========================================================================
    // STEP 6: Apply results to keyframes
    // =========================================================================
    
    // Set initial velocities (will be refined by VIBA)
    // Estimate velocity from position change
    Eigen::Vector3f t1_pos = kf1->GetTwb().block<3,1>(0,3);
    Eigen::Vector3f t2_pos = kf2->GetTwb().block<3,1>(0,3);
    float dt_vel = static_cast<float>(kf2->GetTimestamp() - kf1->GetTimestamp());
    Eigen::Vector3f avg_velocity = Eigen::Vector3f::Zero();
    if (dt_vel > 0.001f) {
        avg_velocity = (t2_pos - t1_pos) / dt_vel;
    }
    
    m_keyframes[0]->SetVelocity(avg_velocity);
    m_keyframes[1]->SetVelocity(avg_velocity);
    
    // Set biases
    for (auto& kf : m_keyframes) {
        kf->SetGyroBias(m_gyro_bias);
        kf->SetAccelBias(Eigen::Vector3f::Zero());  // Accel bias not estimated yet
    }
    
    m_accel_bias = Eigen::Vector3f::Zero();
    m_scale = 1.0;  // Scale already applied in visual init
    
    // Update preintegrator bias
    m_imu_preintegrator->SetBias(m_gyro_bias, m_accel_bias);
    
    if (m_current_frame) {
        m_current_frame->SetGyroBias(m_gyro_bias);
        m_current_frame->SetAccelBias(m_accel_bias);
    }
    
    LOG_INFO("[IMU_INIT_INTERP] IMU initialization SUCCESS!");
    LOG_INFO("  Gravity: [{:.4f}, {:.4f}, {:.4f}]",
             m_gravity.x(), m_gravity.y(), m_gravity.z());
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}]",
             m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
    
    // Apply gravity alignment transformation
    ApplyGravityAlignmentTransform(Rwg.cast<float>(), 1.0f);
    
    // Update gravity to point in -Z direction (after transformation)
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    
    return true;
}

void Estimator::UpdatePreintegrationsWithNewBias(const Eigen::Vector3f& new_gyro_bias,
                                                  const Eigen::Vector3f& new_accel_bias) {
    // Update preintegrations in all keyframes
    for (auto& kf : m_keyframes) {
        auto preint = kf->GetIMUPreintegrationFromLastKeyframe();
        if (preint && preint->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint, new_gyro_bias, new_accel_bias);
        }
        
        auto preint_frame = kf->GetIMUPreintegrationFromLastFrame();
        if (preint_frame && preint_frame->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint_frame, new_gyro_bias, new_accel_bias);
        }
        
        // Update bias in frame
        kf->SetGyroBias(new_gyro_bias);
        kf->SetAccelBias(new_accel_bias);
    }
    
    // Update current frame if exists
    if (m_current_frame) {
        auto preint_frame = m_current_frame->GetIMUPreintegrationFromLastFrame();
        if (preint_frame && preint_frame->IsValid()) {
            m_imu_preintegrator->UpdatePreintegrationWithNewBias(preint_frame, new_gyro_bias, new_accel_bias);
        }
        m_current_frame->SetGyroBias(new_gyro_bias);
        m_current_frame->SetAccelBias(new_accel_bias);
    }
    
    // Update estimator's bias
    m_gyro_bias = new_gyro_bias;
    m_accel_bias = new_accel_bias;
    m_imu_preintegrator->SetBias(new_gyro_bias, new_accel_bias);
}

void Estimator::ApplyGravityAlignmentTransform(const Eigen::Matrix3f& Rwg, float optimized_scale) {
    // Tgw transforms from old world to gravity-aligned world
    // Rgw = Rwg^T (Rwg is world-to-gravity, we need gravity-to-world for inverse)
    Eigen::Matrix3f Rgw = Rwg.transpose();
    
    // Scale factor: optimized_scale means "VO * s = metric"
    // So to convert VO to metric, we apply 1/s
    float scale_factor = 1.0f / optimized_scale;
    
    // 1. Collect all unique MapPoints from keyframes
    std::set<std::shared_ptr<MapPoint>> unique_mps;
    for (const auto& kf : m_keyframes) {
        const auto& features = kf->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            auto mp = kf->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                unique_mps.insert(mp);
            }
        }
    }
    
    // =========================================================================
    // STEP 1: Apply gravity alignment (rotation only)
    // =========================================================================
    
    // Transform MapPoints: p_new = Rgw * p_old
    for (const auto& mp : unique_mps) {
        Eigen::Vector3f pos_old = mp->GetPosition();
        Eigen::Vector3f pos_new = Rgw * pos_old;
        mp->SetPosition(pos_new);
    }
    
    // Transform keyframe poses: R_new = Rgw * R_old, t_new = Rgw * t_old
    for (const auto& kf : m_keyframes) {
        Eigen::Matrix4f Twb_old = kf->GetTwb();
        Eigen::Matrix3f R_old = Twb_old.block<3,3>(0,0);
        Eigen::Vector3f t_old = Twb_old.block<3,1>(0,3);
        
        Eigen::Matrix4f Twb_new = Eigen::Matrix4f::Identity();
        Twb_new.block<3,3>(0,0) = Rgw * R_old;
        Twb_new.block<3,1>(0,3) = Rgw * t_old;
        
        kf->SetTwb(Twb_new);
    }
    
    // Transform velocities: v_new = Rgw * v_old
    for (const auto& kf : m_keyframes) {
        Eigen::Vector3f vel_old = kf->GetVelocity();
        Eigen::Vector3f vel_new = Rgw * vel_old;
        kf->SetVelocity(vel_new);
    }
    
    // =========================================================================
    // STEP 2: Apply scale correction (relative to first keyframe)
    // =========================================================================
    
    if (std::abs(scale_factor - 1.0f) > 1e-6) {
        // Scale keyframe poses relative to first keyframe
        Eigen::Matrix4f Twb_0 = m_keyframes[0]->GetTwb();
        
        for (size_t i = 1; i < m_keyframes.size(); ++i) {
            Eigen::Matrix4f Twb_i = m_keyframes[i]->GetTwb();
            Eigen::Matrix4f T_0_i = Twb_0.inverse() * Twb_i;
            T_0_i.block<3,1>(0,3) *= scale_factor;  // Scale translation relative to first
            m_keyframes[i]->SetTwb(Twb_0 * T_0_i);
        }
        
        // Scale velocities
        for (const auto& kf : m_keyframes) {
            Eigen::Vector3f vel = kf->GetVelocity();
            vel *= scale_factor;
            kf->SetVelocity(vel);
        }
        
        // Scale MapPoints relative to first camera
        Eigen::Matrix4f Twc_0 = m_keyframes[0]->GetTwc();
        Eigen::Matrix4f Tcw_0 = Twc_0.inverse();
        
        for (const auto& mp : unique_mps) {
            Eigen::Vector3f pos_w = mp->GetPosition();
            // Transform to first camera frame
            Eigen::Vector3f pos_c = Tcw_0.block<3,3>(0,0) * pos_w + Tcw_0.block<3,1>(0,3);
            // Apply scale
            pos_c *= scale_factor;
            // Transform back to world
            Eigen::Vector3f pos_w_new = Twc_0.block<3,3>(0,0) * pos_c + Twc_0.block<3,1>(0,3);
            mp->SetPosition(pos_w_new);
        }
    }
    
    // Check if current frame is already in keyframes (same object)
    bool current_in_keyframes = false;
    for (const auto& kf : m_keyframes) {
        if (kf.get() == m_current_frame.get()) {
            current_in_keyframes = true;
            break;
        }
    }
    
    // Update current pose from keyframe if it's the same object
    if (m_current_frame) {
        if (current_in_keyframes) {
            // Current frame is already transformed via keyframes, just update m_current_pose
            m_current_pose = m_current_frame->GetTwb();
        } else {
            // Current frame is different, need to transform separately
            Eigen::Matrix4f Twb_old = m_current_frame->GetTwb();
            Eigen::Matrix3f R_old = Twb_old.block<3,3>(0,0);
            Eigen::Vector3f t_old = Twb_old.block<3,1>(0,3);
            
            Eigen::Matrix4f Twb_new = Eigen::Matrix4f::Identity();
            Twb_new.block<3,3>(0,0) = Rgw * R_old;
            Twb_new.block<3,1>(0,3) = scale_factor * Rgw * t_old;
            
            m_current_frame->SetTwb(Twb_new);
            m_current_pose = Twb_new;
        }
    }
    
    LOG_INFO("Gravity aligned: scale={:.4f}, {} kfs, {} mps", 
             scale_factor, m_keyframes.size(), unique_mps.size());
}

void Estimator::ShrinkWindowAfterInit() {
    // Shrink window from initialization size to tracking size
    // Keep only the most recent m_tracking_window_size keyframes
    
    LOG_INFO("Shrinking window after init: {} -> {} keyframes", 
             m_keyframes.size(), m_tracking_window_size);
    
    // Update window size to tracking size
    m_window_size = m_tracking_window_size;
    
    // Remove oldest keyframes until we reach tracking window size
    while (static_cast<int>(m_keyframes.size()) > m_tracking_window_size) {
        auto oldest_keyframe = m_keyframes.front();
        
        // Save pose before removing from window (for trajectory output)
        m_marginalized_poses.emplace_back(
            oldest_keyframe->GetTimestamp(),
            oldest_keyframe->GetTwb()
        );
        
        // Handle MapPoints whose reference keyframe is being removed
        const auto& old_kf_features = oldest_keyframe->GetFeatures();
        int transferred_count = 0;
        int deleted_count = 0;
        
        for (size_t i = 0; i < old_kf_features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                // Check if this keyframe is the reference for this MapPoint
                if (mp->IsReferenceKeyframe(oldest_keyframe)) {
                    // Find the oldest remaining keyframe that observes this MapPoint
                    std::shared_ptr<Frame> new_ref_keyframe = nullptr;
                    
                    for (size_t kf_idx = 1; kf_idx < m_keyframes.size(); ++kf_idx) {
                        auto& kf = m_keyframes[kf_idx];
                        if (mp->IsObservedByFrame(kf)) {
                            new_ref_keyframe = kf;
                            break;
                        }
                    }
                    
                    if (new_ref_keyframe) {
                        mp->SetReferenceKeyframe(new_ref_keyframe);
                        mp->SetMarginalized(true);
                        transferred_count++;
                    } else {
                        mp->SetBad(true);
                        deleted_count++;
                    }
                }
            }
        }
        
        // Clean up observations from MapPoints
        for (size_t i = 0; i < old_kf_features.size(); ++i) {
            auto mp = oldest_keyframe->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mp->RemoveObservation(oldest_keyframe);
                if (mp->GetObservationCount() == 0) {
                    mp->SetBad(true);
                }
            }
        }
        
        LOG_DEBUG("  Removed kf {}: {} MPs transferred, {} deleted", 
                 oldest_keyframe->GetFrameId(), transferred_count, deleted_count);
        
        m_keyframes.erase(m_keyframes.begin());
    }
    
    // Switch feature tracker to TRACKING settings
    const auto& config = ConfigUtils::GetInstance();
    m_feature_tracker->SetMaxFeatures(config.max_features);
    m_feature_tracker->SetMinDistance(static_cast<float>(config.min_distance));
    m_feature_tracker->SetQualityLevel(static_cast<float>(config.quality_level));
    m_feature_tracker->SetGridParams(config.grid_cols, config.grid_rows, config.max_features_per_grid);
    
    LOG_INFO("Window shrunk: {} keyframes remaining, {} marginalized poses saved",
             m_keyframes.size(), m_marginalized_poses.size());
    LOG_INFO("Feature tracker switched to tracking settings: max_features={}, min_distance={:.1f}",
             config.max_features, config.min_distance);
}

void Estimator::SaveTrajectory(const std::string& output_path) const {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for writing: {}", output_path);
        return;
    }
    
    // Write trajectory in TUM format: timestamp tx ty tz qx qy qz qw
    file << std::fixed << std::setprecision(9);
    
    int saved_count = 0;
    
    // 1. Write marginalized poses (keyframes that left the window)
    for (const auto& [timestamp, T_wb] : m_marginalized_poses) {
        Eigen::Vector3f t = T_wb.block<3, 1>(0, 3);
        Eigen::Matrix3f R = T_wb.block<3, 3>(0, 0);
        Eigen::Quaternionf q(R);
        q.normalize();
        
        file << timestamp << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        saved_count++;
    }
    
    // 2. Write current sliding window keyframes
    for (const auto& kf : m_keyframes) {
        if (!kf) continue;
        
        double timestamp = kf->GetTimestamp();
        Eigen::Matrix4f T_wb = kf->GetTwb();
        
        Eigen::Vector3f t = T_wb.block<3, 1>(0, 3);
        Eigen::Matrix3f R = T_wb.block<3, 3>(0, 0);
        Eigen::Quaternionf q(R);
        q.normalize();
        
        file << timestamp << " "
             << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        saved_count++;
    }
    
    file.close();
    LOG_INFO("Saved trajectory to: {} ({} marginalized + {} window = {} keyframes)", 
             output_path, m_marginalized_poses.size(), m_keyframes.size(), saved_count);
}

} // namespace vio_360
