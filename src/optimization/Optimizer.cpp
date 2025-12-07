/**
 * @file      Optimizer.cpp
 * @brief     Implementation of PnP and Bundle Adjustment optimizers
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Optimizer.h"
#include "Factors.h"
#include "database/Frame.h"
#include "database/MapPoint.h"
#include "database/Feature.h"
#include "database/Camera.h"
#include "processing/IMUPreintegrator.h"
#include "util/Logger.h"
#include "util/LieUtils.h"
#include "util/ConfigUtils.h"

#include <set>

namespace vio_360 {

Optimizer::Optimizer()
    : m_huber_delta(1.0)
    , m_pixel_noise_std(1.0)
    , m_max_iterations(50)
    , m_chi2_threshold(9.21)    // Chi-square 99% for 2 DOF (more permissive)
    , m_camera(nullptr)
    , m_boundary_margin(20) {
}

void Optimizer::SetCamera(std::shared_ptr<Camera> camera, int boundary_margin) {
    m_camera = camera;
    m_boundary_margin = boundary_margin;
}

bool Optimizer::IsNearBoundary(const cv::Point2f& pixel) const {
    if (!m_camera || m_boundary_margin <= 0) {
        return false;
    }
    return m_camera->IsNearBoundary(pixel, static_cast<float>(m_boundary_margin));
}

void Optimizer::PoseToParams(const Eigen::Matrix4f& pose, double* params) {
    // Convert SE3 matrix to tangent space [rho(3), phi(3)] using SE3::log()
    // This is consistent with SE3::exp() used in Factors
    SE3d se3(pose.cast<double>());
    Eigen::Matrix<double, 6, 1> xi = se3.log();
    
    // xi = [rho, phi] where rho is the translation part in tangent space
    params[0] = xi(0);
    params[1] = xi(1);
    params[2] = xi(2);
    params[3] = xi(3);
    params[4] = xi(4);
    params[5] = xi(5);
}

Eigen::Matrix4f Optimizer::ParamsToPose(const double* params) {
    // Convert tangent space [rho(3), phi(3)] to SE3 matrix using SE3::exp()
    // This is consistent with SE3::exp() used in Factors
    Eigen::Matrix<double, 6, 1> xi;
    xi << params[0], params[1], params[2], params[3], params[4], params[5];
    
    SE3d se3 = SE3d::exp(xi);
    return se3.matrix().cast<float>();
}

ceres::Solver::Options Optimizer::SetupSolverOptions(int max_iterations) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = max_iterations;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    return options;
}

PnPResult Optimizer::SolvePnP(std::shared_ptr<Frame> frame, bool fix_mappoints) {
    PnPResult result;
    
    if (!frame) {
        LOG_WARN("SolvePnP: null frame");
        return result;
    }
    
    // Collect observations with valid MapPoints, excluding boundary features
    std::vector<std::tuple<cv::Point2f, std::shared_ptr<MapPoint>, size_t>> observations;
    
    int total_features = 0;
    int invalid_features = 0;
    int no_mappoint = 0;
    int boundary_filtered = 0;
    
    const auto& features = frame->GetFeatures();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        total_features++;
        
        if (!feature || !feature->IsValid()) {
            invalid_features++;
            continue;
        }
        
        auto mp = frame->GetMapPoint(static_cast<int>(i));
        if (!mp || mp->IsBad()) {
            no_mappoint++;
            continue;
        }
        
        // Skip features near horizontal boundary (ERP wrap-around issue)
        if (IsNearBoundary(feature->GetPixelCoord())) {
            boundary_filtered++;
            continue;
        }
        
        observations.push_back({feature->GetPixelCoord(), mp, i});
    }
    
    LOG_DEBUG("  PnP stats: total={}, invalid={}, no_mp={}, boundary={}, valid={}",
              total_features, invalid_features, no_mappoint, boundary_filtered, observations.size());
    
    if (observations.size() < 6) {
        LOG_WARN("SolvePnP: insufficient observations ({})", observations.size());
        return result;
    }
    
    // Equirectangular camera parameters (cols, rows)
    double cols = static_cast<double>(frame->GetWidth());
    double rows = static_cast<double>(frame->GetHeight());
    factor::CameraParameters cam_params(cols, rows);
    
    // Get body-to-camera transform
    Eigen::Matrix4d T_cb = frame->GetTCB().cast<double>();
    
    // Information matrix
    Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
    
    // Store initial pose (for right perturbation approach)
    Eigen::Matrix4f T_wb_init = frame->GetTwb();
    Eigen::Matrix4d T_wb_init_d = T_wb_init.cast<double>();
    
    // Log initial pose for debugging
    Eigen::Vector3f init_pos = T_wb_init.block<3,1>(0,3);
    LOG_DEBUG("  [PnP] Frame {} init pose: ({:.4f},{:.4f},{:.4f}), {} observations",
              frame->GetFrameId(), init_pos.x(), init_pos.y(), init_pos.z(), observations.size());
    
    // Setup pose parameters as perturbation (initialized to zero)
    // Right perturbation: T_wb = T_wb_init * exp(delta_xi)
    double pose_params[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::PnPFactor*> factors;
    std::vector<size_t> feature_indices;
    std::vector<std::shared_ptr<MapPoint>> mappoints;  // Store MapPoints for marginalized check
    
    for (const auto& [pixel_coord, mp, feat_idx] : observations) {
        Eigen::Vector2d obs(pixel_coord.x, pixel_coord.y);
        Eigen::Vector3d world_pt = mp->GetPosition().cast<double>();
        
        // Pass initial pose to factor for right perturbation
        auto* cost = new factor::PnPFactor(obs, world_pt, cam_params, T_cb, T_wb_init_d, info);
        factors.push_back(cost);
        feature_indices.push_back(feat_idx);
        mappoints.push_back(mp);
        
        // Marginalized MapPoints get 2x weight (scale preserving)
        double weight = mp->IsMarginalized() ? 2.0 : 1.0;
        ceres::LossFunction* loss = new ceres::HuberLoss(m_huber_delta);
        if (weight > 1.0) {
            loss = new ceres::ScaledLoss(loss, weight, ceres::TAKE_OWNERSHIP);
        }
        
        problem.AddResidualBlock(
            cost,
            loss,
            pose_params
        );
    }
    
    // 4-round outlier detection
    const int num_rounds = 4;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int total_iterations = 0;
    
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    
    for (int round = 0; round < num_rounds; ++round) {
        // Reset perturbation to zero for each round (right perturbation approach)
        if (round > 0) {
            std::fill(pose_params, pose_params + 6, 0.0);
        }
        
        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        if (round == 0) {
            initial_cost = summary.initial_cost;
        }
        final_cost = summary.final_cost;
        total_iterations += summary.iterations.size();
        
        // Bearing-based outlier detection for equirectangular cameras
        // Threshold: 2 degrees = 0.035 radians
        const double bearing_threshold = 2.0 * M_PI / 180.0;  // 2 degrees in radians
        
        const double* params_ptr = pose_params;
        int num_inliers = 0;
        int num_outliers = 0;
        double inlier_reproj_sum = 0.0;
        double max_chi2 = 0.0;
        double max_inlier_chi2 = 0.0;
        int chi2_bins[6] = {0};  // [0-2], [2-6], [6-10], [10-50], [50-100], [100+]
        
        for (size_t i = 0; i < factors.size(); ++i) {
            double chi2 = factors[i]->compute_chi_square(&params_ptr);
            
            // Marginalized MapPoints should never be marked as outliers (they preserve scale)
            bool is_marginalized = mappoints[i]->IsMarginalized();
            bool is_outlier = !is_marginalized && (chi2 > 5.991);
            
            // Track max chi2
            if (chi2 > max_chi2) max_chi2 = chi2;
            
            // Bin distribution
            if (chi2 < 2.0) chi2_bins[0]++;
            else if (chi2 < 6.0) chi2_bins[1]++;
            else if (chi2 < 10.0) chi2_bins[2]++;
            else if (chi2 < 50.0) chi2_bins[3]++;
            else if (chi2 < 100.0) chi2_bins[4]++;
            else chi2_bins[5]++;
            
            factors[i]->set_outlier(is_outlier);
            
            if (is_outlier) {
                num_outliers++;
            } else {
                num_inliers++;
                // Convert bearing error to approximate pixel error for logging
                // (bearing_error * image_width / (2*pi) gives rough pixel equivalent)
                inlier_reproj_sum += chi2;  // degrees for logging
                if (chi2 > max_inlier_chi2) max_inlier_chi2 = chi2;
            }
        }
        
        // Debug log for error distribution on final round
        if (round == num_rounds - 1) {
            LOG_DEBUG("  [PnP] chi2 dist: [0-2]={} [2-6]={} [6-10]={} [10-50]={} [50-100]={} [100+]={}",
                     chi2_bins[0], chi2_bins[1], chi2_bins[2], chi2_bins[3], chi2_bins[4], chi2_bins[5]);
            LOG_DEBUG("  [PnP] max_chi2={:.1f}, max_inlier_chi2={:.1f}, mean_inlier={:.2f}",
                     max_chi2, max_inlier_chi2, num_inliers > 0 ? inlier_reproj_sum / num_inliers : 0.0);
        }
        
        // Update result
        result.num_inliers = num_inliers;
        result.num_outliers = num_outliers;
        result.success = summary.IsSolutionUsable();
        
        // Compute mean inlier reprojection error
        if (num_inliers > 0) {
            final_cost = inlier_reproj_sum / num_inliers;
        }
        
        // Debug log for each round
        LOG_DEBUG("  PnP round {}: inliers={}, outliers={}, cost={:.2f}", 
                  round, num_inliers, num_outliers, final_cost);
    }
    
    // Compute final pose: T_wb = T_wb_init * exp(delta_xi) (right perturbation)
    Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params);
    SE3d T_wb_init_se3(T_wb_init_d);
    SE3d delta_T = SE3d::exp(delta_xi);
    SE3d T_wb_final = T_wb_init_se3 * delta_T;
    result.optimized_pose = T_wb_final.matrix().cast<float>();
    
    // Log pose change for debugging
    Eigen::Matrix4f old_pose = T_wb_init;
    Eigen::Vector3f old_pos = old_pose.block<3,1>(0,3);
    Eigen::Vector3f new_pos = result.optimized_pose.block<3,1>(0,3);
    float pos_change = (new_pos - old_pos).norm();
    
    // Safety check: reject if too few inliers or too large pose change
    const int min_inliers_for_update = 10;
    const float max_pose_change = 0.5f;  // 0.5m threshold
    
    if (result.num_inliers < min_inliers_for_update) {
        LOG_WARN("  PnP rejected: too few inliers ({} < {}), keeping predicted pose",
                 result.num_inliers, min_inliers_for_update);
        result.success = false;
        result.optimized_pose = old_pose;  // Keep old pose
    } 
     
    else {
        frame->SetTwb(result.optimized_pose);
    }
    
    result.initial_cost = initial_cost;
    result.final_cost = final_cost;  // Now this is mean inlier reprojection error in pixels
    result.num_iterations = total_iterations;
    
    return result;
}

BAResult Optimizer::RunBA(const std::vector<std::shared_ptr<Frame>>& frames,
                          bool fix_first_pose,
                          bool fix_last_pose) {
    BAResult result;
    
    if (frames.size() < 2) {
        LOG_WARN("RunBA: need at least 2 frames");
        return result;
    }
    
    // Collect all MapPoints observed by these frames
    std::set<std::shared_ptr<MapPoint>> mappoint_set;
    for (const auto& frame : frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mappoint_set.insert(mp);
            }
        }
    }
    
    std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
    if (mappoints.empty()) {
        LOG_WARN("RunBA: no valid MapPoints");
        return result;
    }
    
    // Parameter blocks for right perturbation
    std::vector<Eigen::Matrix4d> T_wb_inits(frames.size());  // Store initial poses
    std::vector<std::array<double, 6>> pose_params(frames.size());  // delta_xi (perturbation from init)
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize: delta_xi = 0 (right perturbation), store T_wb_init
    for (size_t i = 0; i < frames.size(); ++i) {
        T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // delta_xi starts at 0
    }
    
    // Initialize point parameters and create index map
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::BAFactor*> factors;
    std::vector<std::pair<size_t, size_t>> factor_indices;  // (frame_idx, mp_idx)
    
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const auto& frame = frames[fi];
        
        // Equirectangular camera parameters (cols, rows)
        double cols = static_cast<double>(frame->GetWidth());
        double rows = static_cast<double>(frame->GetHeight());
        factor::CameraParameters cam_params(cols, rows);
        
        Eigen::Matrix4d T_cb = frame->GetTCB().cast<double>();
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
        
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (!mp || mp->IsBad()) continue;
            
            // Skip features near horizontal boundary (ERP wrap-around issue)
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            auto it = mp_to_idx.find(mp);
            if (it == mp_to_idx.end()) continue;
            
            size_t pi = it->second;
            Eigen::Vector2d obs(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            // Right perturbation: pass T_wb_init to factor
            auto* cost = new factor::BAFactor(obs, cam_params, T_cb, T_wb_inits[fi], info);
            factors.push_back(cost);
            factor_indices.push_back({fi, pi});
            
            // Marginalized MapPoints get 2x weight (scale preserving)
            double weight = mp->IsMarginalized() ? 2.0 : 1.0;
            ceres::LossFunction* loss = new ceres::HuberLoss(m_huber_delta);
            if (weight > 1.0) {
                loss = new ceres::ScaledLoss(loss, weight, ceres::TAKE_OWNERSHIP);
            }
            
            problem.AddResidualBlock(
                cost,
                loss,
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Fix poses as requested
    if (fix_first_pose && !frames.empty()) {
        problem.SetParameterBlockConstant(pose_params[0].data());
    }
    if (fix_last_pose && frames.size() > 1) {
        problem.SetParameterBlockConstant(pose_params.back().data());
    }
    
    // Solve
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Bearing-based outlier detection for equirectangular cameras
    const double bearing_threshold = 2.0 * M_PI / 180.0;  // 2 degrees
    
    int num_inliers = 0;
    int num_outliers = 0;
    
    // Count MapPoint outliers using bearing angle error
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        size_t fi = factor_indices[i].first;
        size_t pi = factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > 5.991);
        factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark MapPoints as bad if ALL observations are outliers (more conservative)
    // But NEVER mark marginalized MapPoints as bad (they preserve scale)
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;  // Marginalized MapPoints must not be removed
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        // Only mark bad if no inliers at all and at least 2 outlier observations
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    
    // Update frames and MapPoints
    // Right Perturbation: T_wb = T_wb_init * exp(delta_xi)
    for (size_t i = 0; i < frames.size(); ++i) {
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
        vio_360::SE3d T_wb_init_se3(T_wb_inits[i]);
        vio_360::SE3d T_wb_final = T_wb_init_se3 * delta_T;
        frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
    }
    
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsBad() && !mappoints[i]->IsMarginalized()) {
            Eigen::Vector3f pos(point_params[i][0], point_params[i][1], point_params[i][2]);
            mappoints[i]->SetPosition(pos);
        }
    }
    
    result.success = summary.IsSolutionUsable();
    result.num_inliers = num_inliers;
    result.num_outliers = num_outliers;
    result.num_poses_optimized = frames.size();
    result.num_points_optimized = mappoints.size();
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();
    
    return result;
}

BAResult Optimizer::RunFullBA(const std::vector<std::shared_ptr<Frame>>& frames) {
    // Full BA: fix only first pose, optimize second pose and all MapPoints
    return RunBA(frames, true, false);
}

BAResult Optimizer::RunLocalBAwithInertial(const std::vector<std::shared_ptr<Frame>>& window_frames,
                                            const Eigen::Vector3f& gravity) {
    BAResult result;
    
    if (window_frames.size() < 2) {
        LOG_WARN("RunLocalBAwithInertial: need at least 2 frames in window");
        return result;
    }
    
    // LOG_INFO("Starting Local BA with Inertial factors ({} frames)", window_frames.size());
    
    // ==================== Step 1: Collect MapPoints ====================
    std::set<std::shared_ptr<MapPoint>> mappoint_set;
    for (const auto& frame : window_frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mappoint_set.insert(mp);
            }
        }
    }
    
    std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
    if (mappoints.empty()) {
        LOG_WARN("RunLocalBAwithInertial: no valid MapPoints");
        return result;
    }
    
    // Create frame index map
    std::map<std::shared_ptr<Frame>, size_t> frame_to_idx;
    for (size_t i = 0; i < window_frames.size(); ++i) {
        frame_to_idx[window_frames[i]] = i;
    }
    
    // ==================== Step 2: Parameter blocks ====================
    // Pose parameters (6D perturbation, right perturbation: T_wb = T_wb_init * exp(delta_xi))
    std::vector<Eigen::Matrix4d> T_wb_inits(window_frames.size());
    std::vector<std::array<double, 6>> pose_params(window_frames.size());
    
    // Velocity parameters (3D per frame)
    std::vector<std::array<double, 3>> velocity_params(window_frames.size());
    
    // Bias parameters (PER FRAME - individual biases)
    std::vector<std::array<double, 3>> gyro_bias_params(window_frames.size());
    std::vector<std::array<double, 3>> accel_bias_params(window_frames.size());
    
    // Store initial values for priors
    std::vector<Eigen::Vector3d> velocity_priors(window_frames.size());
    std::vector<Eigen::Vector3d> gyro_bias_priors(window_frames.size());
    std::vector<Eigen::Vector3d> accel_bias_priors(window_frames.size());
    
    // Point parameters
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize parameters from each frame
    for (size_t i = 0; i < window_frames.size(); ++i) {
        T_wb_inits[i] = window_frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // delta_xi starts at 0
        
        Eigen::Vector3f vel = window_frames[i]->GetVelocity();
        velocity_params[i] = {vel.x(), vel.y(), vel.z()};
        velocity_priors[i] = vel.cast<double>();
        
        Eigen::Vector3f gb = window_frames[i]->GetGyroBias();
        Eigen::Vector3f ab = window_frames[i]->GetAccelBias();
        gyro_bias_params[i] = {gb.x(), gb.y(), gb.z()};
        accel_bias_params[i] = {ab.x(), ab.y(), ab.z()};
        gyro_bias_priors[i] = gb.cast<double>();
        accel_bias_priors[i] = ab.cast<double>();
    }
    
    // Initialize point parameters
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // ==================== Step 3: Build optimization problem ====================
    ceres::Problem problem;
    std::vector<factor::BAFactor*> ba_factors;
    std::vector<std::pair<size_t, size_t>> ba_factor_indices;  // (frame_idx, mp_idx)
    
    // Add visual factors (BAFactor with right perturbation)
    for (size_t pi = 0; pi < mappoints.size(); ++pi) {
        const auto& mp = mappoints[pi];
        const auto& observations = mp->GetObservations();
        
        for (const auto& obs : observations) {
            auto obs_frame = obs.frame.lock();
            if (!obs_frame) continue;
            
            auto frame_it = frame_to_idx.find(obs_frame);
            if (frame_it == frame_to_idx.end()) continue;
            
            size_t fi = frame_it->second;
            
            const auto& features = obs_frame->GetFeatures();
            if (obs.feature_index < 0 || obs.feature_index >= static_cast<int>(features.size())) continue;
            
            const auto& feature = features[obs.feature_index];
            if (!feature || !feature->IsValid()) continue;
            
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            double cols = static_cast<double>(obs_frame->GetWidth());
            double rows = static_cast<double>(obs_frame->GetHeight());
            factor::CameraParameters cam_params(cols, rows);
            
            Eigen::Matrix4d T_cb = obs_frame->GetTCB().cast<double>();
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
            
            Eigen::Vector2d obs_pixel(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            auto* cost = new factor::BAFactor(obs_pixel, cam_params, T_cb, T_wb_inits[fi], info);
            ba_factors.push_back(cost);
            ba_factor_indices.push_back({fi, pi});
            
            // Marginalized MapPoints get 2x weight (scale preserving)
            double weight = mp->IsMarginalized() ? 2.0 : 1.0;
            ceres::LossFunction* loss = new ceres::HuberLoss(m_huber_delta);
            if (weight > 1.0) {
                loss = new ceres::ScaledLoss(loss, weight, ceres::TAKE_OWNERSHIP);
            }
            
            problem.AddResidualBlock(
                cost,
                loss,
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Add IMU factors between consecutive keyframes (InertialFactorFixedGravity)
    // Uses individual biases per frame (8 parameter blocks)
    Eigen::Vector3d gravity_d = gravity.cast<double>();
    int imu_factor_count = 0;
    
    for (size_t i = 0; i < window_frames.size() - 1; ++i) {
        auto preint = window_frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
        if (!preint) {
            LOG_WARN("RunLocalBAwithInertial: No IMU preintegration for frame {}", i + 1);
            continue;
        }
        
        auto* imu_cost = new factor::InertialFactorFixedGravity(
            preint, gravity_d, T_wb_inits[i], T_wb_inits[i + 1]);
        
        // Pass both frame_i and frame_j bias (8 parameter blocks)
        // Preintegration uses frame_i's bias for correction
        problem.AddResidualBlock(
            imu_cost,
            nullptr,  // No robust loss for IMU
            pose_params[i].data(),
            velocity_params[i].data(),
            gyro_bias_params[i].data(),      // bias for frame i
            accel_bias_params[i].data(),     // bias for frame i
            pose_params[i + 1].data(),
            velocity_params[i + 1].data(),
            gyro_bias_params[i + 1].data(),  // bias for frame j
            accel_bias_params[i + 1].data()  // bias for frame j
        );
        imu_factor_count++;
    }
    
    LOG_DEBUG("  Added {} IMU factors", imu_factor_count);
    
    // Add bias consistency factors between consecutive frames
    // This ensures bias doesn't change too rapidly between frames
    const double bias_consistency_weight = 100.0;  // Very strong consistency constraint
    for (size_t i = 0; i < window_frames.size() - 1; ++i) {
        // Gyro bias consistency
        auto* gyro_consistency = new factor::BiasConsistencyFactor(bias_consistency_weight);
        problem.AddResidualBlock(gyro_consistency, nullptr,
                                gyro_bias_params[i].data(),
                                gyro_bias_params[i + 1].data());
        
        // Accel bias consistency
        auto* accel_consistency = new factor::BiasConsistencyFactor(bias_consistency_weight);
        problem.AddResidualBlock(accel_consistency, nullptr,
                                accel_bias_params[i].data(),
                                accel_bias_params[i + 1].data());
    }
    
    LOG_DEBUG("  Added {} bias consistency factors", 2 * (window_frames.size() - 1));
    
    // Add velocity and bias priors with UNIFORM strong weight for all frames
    const size_t n_frames = window_frames.size();
    const double velocity_prior_weight = 10.0;   // Moderate velocity prior
    const double bias_prior_weight = 1000.0;     // Very strong bias prior to prevent large jumps
    
    for (size_t i = 0; i < n_frames; ++i) {
        // Velocity prior
        auto* vel_prior = new factor::BiasPriorFactor(velocity_priors[i], velocity_prior_weight);
        problem.AddResidualBlock(vel_prior, nullptr, velocity_params[i].data());
        
        // Gyro bias prior
        auto* gyro_prior = new factor::BiasPriorFactor(gyro_bias_priors[i], bias_prior_weight);
        problem.AddResidualBlock(gyro_prior, nullptr, gyro_bias_params[i].data());
        
        // Accel bias prior
        auto* accel_prior = new factor::BiasPriorFactor(accel_bias_priors[i], bias_prior_weight);
        problem.AddResidualBlock(accel_prior, nullptr, accel_bias_params[i].data());
    }
    
    LOG_DEBUG("  Added priors: uniform weight vel={:.2f} bias={:.2f} for all {} frames",
             velocity_prior_weight, bias_prior_weight, n_frames);
    
    // ==================== Step 4: Fix constraints ====================
    std::set<size_t> poses_in_problem;
    for (const auto& [fi, pi] : ba_factor_indices) {
        poses_in_problem.insert(fi);
    }
    for (size_t i = 0; i < window_frames.size(); ++i) {
        poses_in_problem.insert(i);
    }
    
    // Fix first keyframe pose for scale stability
    constexpr size_t NUM_FIXED_KEYFRAMES = 1;
    int fixed_count = 0;
    int optimized_count = 0;
    
    for (size_t i = 0; i < window_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;
        
        if (i < NUM_FIXED_KEYFRAMES) {
            problem.SetParameterBlockConstant(pose_params[i].data());
            fixed_count++;
        } else {
            optimized_count++;
        }
    }
    
    // Fix marginalized MapPoints
    int fixed_mp_count = 0;
    int optimized_mp_count = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (mappoints[i]->IsMarginalized()) {
            problem.SetParameterBlockConstant(point_params[i].data());
            fixed_mp_count++;
        } else {
            optimized_mp_count++;
        }
    }
    
    LOG_DEBUG("  LocalBA+I: {} frames (fix {}, opt {}), {} MPs ({} fixed, {} opt), {} visual, {} IMU",
             window_frames.size(), fixed_count, optimized_count, 
             mappoints.size(), fixed_mp_count, optimized_mp_count, 
             ba_factors.size(), imu_factor_count);
    
    // ==================== Step 5: Solve ====================
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // ==================== Step 6: Outlier detection ====================
    constexpr double chi_sq_threshold = 5.99146;
    int num_inliers = 0;
    int num_outliers = 0;
    
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < ba_factors.size(); ++i) {
        size_t fi = ba_factor_indices[i].first;
        size_t pi = ba_factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = ba_factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > chi_sq_threshold);
        ba_factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark bad MapPoints
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    if (bad_mp_count > 0) {
        LOG_DEBUG("  LocalBA+I: marked {} MapPoints as bad", bad_mp_count);
    }
    
    // ==================== Step 7: Update results ====================
    // Update poses (Right Perturbation: T_wb = T_wb_init * exp(delta_xi))
    for (size_t i = NUM_FIXED_KEYFRAMES; i < window_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;
        
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
        vio_360::SE3d T_wb_init_se3(T_wb_inits[i]);
        vio_360::SE3d T_wb_final = T_wb_init_se3 * delta_T;
        window_frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
    }
    
    // Update velocities
    for (size_t i = 0; i < window_frames.size(); ++i) {
        Eigen::Vector3f new_vel(velocity_params[i][0], velocity_params[i][1], velocity_params[i][2]);
        window_frames[i]->SetVelocity(new_vel);
    }
    
    // Update biases (individual per frame)
    for (size_t i = 0; i < window_frames.size(); ++i) {
        Eigen::Vector3f new_gyro_bias(gyro_bias_params[i][0], gyro_bias_params[i][1], gyro_bias_params[i][2]);
        Eigen::Vector3f new_accel_bias(accel_bias_params[i][0], accel_bias_params[i][1], accel_bias_params[i][2]);
        window_frames[i]->SetGyroBias(new_gyro_bias);
        window_frames[i]->SetAccelBias(new_accel_bias);
    }
    
    // Update MapPoints
    int newly_marginalized = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsMarginalized()) {
            Eigen::Vector3f new_pos(point_params[i][0], point_params[i][1], point_params[i][2]);
            mappoints[i]->SetPosition(new_pos);
            
            // Increment BA count and marginalize if optimized at least once
            mappoints[i]->IncrementBACount();
            if (mappoints[i]->GetBACount() >= 1) {
                mappoints[i]->SetMarginalized(true);
                newly_marginalized++;
            }
        }
    }
    
    if (newly_marginalized > 0) {
        LOG_DEBUG("  LocalBA+I: {} MapPoints marginalized (BA count >= 1)", newly_marginalized);
    }
    
    result.success = summary.IsSolutionUsable();
    result.num_inliers = num_inliers;
    result.num_outliers = num_outliers;
    result.num_poses_optimized = window_frames.size() - fixed_count;
    result.num_points_optimized = optimized_mp_count;
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();
    
    return result;
}

// BAResult Optimizer::RunVIBA(const std::vector<std::shared_ptr<Frame>>& frames,
//                             bool fix_first_pose) {
//     BAResult result;


//     LOG_INFO("Starting Visual-Inertial Bundle Adjustment (VIBA) with {} frames", frames.size());
    
//     if (frames.size() < 2) {
//         LOG_WARN("RunVIBA: need at least 2 frames");
//         return result;
//     }
    
//     // Collect all MapPoints observed by these frames
//     std::set<std::shared_ptr<MapPoint>> mappoint_set;
//     for (const auto& frame : frames) {
//         const auto& features = frame->GetFeatures();
//         for (size_t i = 0; i < features.size(); ++i) {
//             const auto& feature = features[i];
//             if (!feature || !feature->IsValid()) continue;
//             auto mp = frame->GetMapPoint(static_cast<int>(i));
//             if (mp && !mp->IsBad()) {
//                 mappoint_set.insert(mp);
//             }
//         }
//     }
    
//     std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
//     if (mappoints.empty()) {
//         LOG_WARN("RunVIBA: no valid MapPoints");
//         return result;
//     }
    
//     // ==================== Parameter blocks ====================
//     // Pose parameters (6D perturbation, right perturbation: T_wb = T_wb_init * exp(delta_xi))
//     std::vector<Eigen::Matrix4d> T_wb_inits(frames.size());  // Store initial poses
//     std::vector<std::array<double, 6>> pose_params(frames.size());  // delta_xi (perturbation)
    
//     // Velocity parameters (3D each frame)
//     std::vector<std::array<double, 3>> velocity_params(frames.size());
    
//     // Shared bias parameters (gyro and accel - same for all frames in VIBA)
//     std::array<double, 3> gyro_bias_params;
//     std::array<double, 3> accel_bias_params;
    
//     // Gravity direction (2D) and scale (1D) - fixed after IMU init
//     std::array<double, 2> gravity_dir_params = {0.0, 0.0};
//     std::array<double, 1> scale_params = {1.0};
    
//     // Point parameters
//     std::vector<std::array<double, 3>> point_params(mappoints.size());
    
//     // Initialize pose, velocity, and bias parameters from each frame
//     for (size_t i = 0; i < frames.size(); ++i) {
//         // Pose: Right perturbation - store T_wb_init, delta_xi = 0
//         T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
//         pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // delta_xi starts at 0
        
//         // Velocity
//         Eigen::Vector3f vel = frames[i]->GetVelocity();
//         velocity_params[i] = {vel.x(), vel.y(), vel.z()};
        
//         // Note: bias is shared, initialized from last frame (most recent estimate)
//         if (i == frames.size() - 1) {
//             Eigen::Vector3f gb = frames[i]->GetGyroBias();
//             Eigen::Vector3f ab = frames[i]->GetAccelBias();
//             gyro_bias_params = {gb.x(), gb.y(), gb.z()};
//             accel_bias_params = {ab.x(), ab.y(), ab.z()};
//         }
//     }
    
//     // Gravity direction is fixed after IMU init, so just use default (0, 0)
//     // This corresponds to gravity = [0, 0, -9.81] in world frame
//     // The actual gravity direction was already applied during IMU initialization
//     // gravity_dir_params stays at {0.0, 0.0}
    
//     // Initialize point parameters
//     std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
//     for (size_t i = 0; i < mappoints.size(); ++i) {
//         Eigen::Vector3f pos = mappoints[i]->GetPosition();
//         point_params[i] = {pos.x(), pos.y(), pos.z()};
//         mp_to_idx[mappoints[i]] = i;
//     }
    
//     // ==================== STAGE 1: Optimize last frame's velocity and bias ====================
//     // Add ALL IMU factors between consecutive frames
//     // Fix all poses, all velocities/biases except last frame's velocity and bias
//     // Only optimize: velocity[last_idx] and gyro_bias[last_idx], accel_bias[last_idx]
//     {
//         ceres::Problem problem_stage1;
        
//         size_t last_idx = frames.size() - 1;
//         const auto& config = ConfigUtils::GetInstance();
        
//         // Add ALL IMU factors between consecutive frames (right perturbation)
//         for (size_t i = 0; i < frames.size() - 1; ++i) {
//             auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
//             if (!preint) continue;
            
//             auto* imu_cost = new factor::InertialGravityScaleFactor(
//                 preint, config.imu_gravity_magnitude);
            
//             problem_stage1.AddResidualBlock(
//                 imu_cost,
//                 nullptr,
//                 pose_params[i].data(),
//                 velocity_params[i].data(),
//                 gyro_bias_params.data(),
//                 accel_bias_params.data(),
//                 pose_params[i + 1].data(),
//                 velocity_params[i + 1].data(),
//                 gravity_dir_params.data(),
//                 scale_params.data()
//             );
//         }
        
//         // Fix ALL poses
//         for (size_t i = 0; i < frames.size(); ++i) {
//             problem_stage1.SetParameterBlockConstant(pose_params[i].data());
//         }
        
//         // Fix ALL velocities except last frame
//         for (size_t i = 0; i < frames.size() - 1; ++i) {
//             problem_stage1.SetParameterBlockConstant(velocity_params[i].data());
//         }
//         // velocity_params[last_idx] is FREE
        
//         // Shared bias is FREE in Stage 1 (will be optimized)
        
//         // Fix gravity and scale
//         problem_stage1.SetParameterBlockConstant(gravity_dir_params.data());
//         problem_stage1.SetParameterBlockConstant(scale_params.data());
        
//         // Add bias prior to prevent drift (for shared bias)
//         Eigen::Vector3d gyro_prior_val(gyro_bias_params[0], 
//                                         gyro_bias_params[1], 
//                                         gyro_bias_params[2]);
//         Eigen::Vector3d accel_prior_val(accel_bias_params[0], 
//                                          accel_bias_params[1], 
//                                          accel_bias_params[2]);
//         const double bias_prior_weight = 1e1;
//         auto* gyro_prior = new factor::BiasPriorFactor(gyro_prior_val, bias_prior_weight);
//         auto* accel_prior = new factor::BiasPriorFactor(accel_prior_val, bias_prior_weight);
//         problem_stage1.AddResidualBlock(gyro_prior, nullptr, gyro_bias_params.data());
//         problem_stage1.AddResidualBlock(accel_prior, nullptr, accel_bias_params.data());
        
//         ceres::Solver::Options opts1;
//         opts1.linear_solver_type = ceres::DENSE_QR;
//         opts1.max_num_iterations = 20;
//         opts1.minimizer_progress_to_stdout = false;
//         ceres::Solver::Summary summary1;
//         ceres::Solve(opts1, &problem_stage1, &summary1);
        
//         LOG_INFO("VIBA Stage1: iter={}, cost {:.4f} -> {:.4f}", 
//                  summary1.iterations.size(), summary1.initial_cost, summary1.final_cost);
//     }
    
//     // Update last frame's velocity and shared bias from Stage 1
//     size_t last_idx = frames.size() - 1;
//     Eigen::Vector3f new_vel(velocity_params[last_idx][0], velocity_params[last_idx][1], velocity_params[last_idx][2]);
//     frames[last_idx]->SetVelocity(new_vel);
    
//     // Update optimized shared bias to all frames
//     Eigen::Vector3f optimized_gyro_bias(gyro_bias_params[0], 
//                                          gyro_bias_params[1], 
//                                          gyro_bias_params[2]);
//     Eigen::Vector3f optimized_accel_bias(accel_bias_params[0], 
//                                           accel_bias_params[1], 
//                                           accel_bias_params[2]);

//     // Update bias to all frames (shared bias)
//     for (size_t i = 0; i < frames.size(); ++i) {
//         frames[i]->SetGyroBias(optimized_gyro_bias);
//         frames[i]->SetAccelBias(optimized_accel_bias);
//     }

    

//     LOG_INFO("VIBA Stage1: velocity=[{:.3f},{:.3f},{:.3f}], bias bg=[{:.6f},{:.6f},{:.6f}] ba=[{:.6f},{:.6f},{:.6f}]",
//              new_vel.x(), new_vel.y(), new_vel.z(),
//              optimized_gyro_bias.x(), optimized_gyro_bias.y(), optimized_gyro_bias.z(),
//              optimized_accel_bias.x(), optimized_accel_bias.y(), optimized_accel_bias.z());
    
//     // ==================== STAGE 2: Visual BA only ====================
//     // Fix: first pose, all velocities, all biases, gravity, scale
//     // Optimize: poses (except first), mappoints
//     ceres::Problem problem;
//     std::vector<factor::BAFactor*> ba_factors;
//     std::vector<std::pair<size_t, size_t>> ba_factor_indices;  // (frame_idx, mp_idx)
    
//     // Add visual factors (BAFactor) - Right Perturbation
//     for (size_t fi = 0; fi < frames.size(); ++fi) {
//         const auto& frame = frames[fi];
        
//         double cols = static_cast<double>(frame->GetWidth());
//         double rows = static_cast<double>(frame->GetHeight());
//         factor::CameraParameters cam_params(cols, rows);
        
//         Eigen::Matrix4d T_cb = frame->GetTCB().cast<double>();
//         Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
        
//         const auto& features = frame->GetFeatures();
//         for (size_t i = 0; i < features.size(); ++i) {
//             const auto& feature = features[i];
//             if (!feature || !feature->IsValid()) continue;
            
//             auto mp = frame->GetMapPoint(static_cast<int>(i));
//             if (!mp || mp->IsBad()) continue;
            
//             if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
//             auto it = mp_to_idx.find(mp);
//             if (it == mp_to_idx.end()) continue;
            
//             size_t pi = it->second;
//             Eigen::Vector2d obs(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
//             // Right perturbation: pass T_wb_init to factor
//             auto* cost = new factor::BAFactor(obs, cam_params, T_cb, T_wb_inits[fi], info);
//             ba_factors.push_back(cost);
//             ba_factor_indices.push_back({fi, pi});
            
//             problem.AddResidualBlock(
//                 cost,
//                 new ceres::HuberLoss(m_huber_delta),
//                 pose_params[fi].data(),
//                 point_params[pi].data()
//             );
//         }
//     }
    
//     // Add ALL IMU factors to Stage 2 (right perturbation)
//     const auto& config = ConfigUtils::GetInstance();
//     for (size_t i = 0; i < frames.size() - 1; ++i) {
//         auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
//         if (!preint) continue;
        
//         auto* imu_cost = new factor::InertialGravityScaleFactor(
//             preint, config.imu_gravity_magnitude);
        
//         problem.AddResidualBlock(
//             imu_cost,
//             nullptr,
//             pose_params[i].data(),
//             velocity_params[i].data(),
//             gyro_bias_params.data(),
//             accel_bias_params.data(),
//             pose_params[i + 1].data(),
//             velocity_params[i + 1].data(),
//             gravity_dir_params.data(),
//             scale_params.data()
//         );
//     }
    
//     // Stage 2: Visual-Inertial BA
//     // Fix first pose only, optimize other poses, mappoints, velocities and biases
//     if (fix_first_pose && !frames.empty()) {
//         problem.SetParameterBlockConstant(pose_params[0].data());
//     }
    
//     // Add velocity priors from Stage 1 (regularization, not fixed)
//     const double velocity_prior_weight = 1000.0;
//     for (size_t i = 0; i < frames.size(); ++i) {
//         Eigen::Vector3d vel_prior(velocity_params[i][0], velocity_params[i][1], velocity_params[i][2]);
//         auto* vel_prior_factor = new factor::BiasPriorFactor(vel_prior, velocity_prior_weight);
//         problem.AddResidualBlock(vel_prior_factor, nullptr, velocity_params[i].data());
//     }
    
//     // Add bias priors from Stage 1 (regularization, not fixed)
//     const double bias_prior_weight = 100.0;
//     Eigen::Vector3d gyro_bias_prior(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
//     Eigen::Vector3d accel_bias_prior(accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
//     auto* gyro_prior = new factor::BiasPriorFactor(gyro_bias_prior, bias_prior_weight);
//     auto* accel_prior = new factor::BiasPriorFactor(accel_bias_prior, bias_prior_weight);
//     problem.AddResidualBlock(gyro_prior, nullptr, gyro_bias_params.data());
//     problem.AddResidualBlock(accel_prior, nullptr, accel_bias_params.data());
    
//     // Fix gravity and scale (already determined in IMU init)
//     problem.SetParameterBlockConstant(gravity_dir_params.data());
//     problem.SetParameterBlockConstant(scale_params.data());  // Scale is fixed
    
//     // ==================== Solve Stage 2 ====================
//     // Log before Stage 2 for verification
//     LOG_INFO("VIBA Stage2 BEFORE: vel[last]=[{:.3f},{:.3f},{:.3f}], bg=[{:.6f},{:.6f},{:.6f}], ba=[{:.6f},{:.6f},{:.6f}], scale={:.6f}",
//              velocity_params[last_idx][0], velocity_params[last_idx][1], velocity_params[last_idx][2],
//              gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2],
//              accel_bias_params[0], accel_bias_params[1], accel_bias_params[2],
//              scale_params[0]);
    
//     ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
//     options.linear_solver_type = ceres::SPARSE_SCHUR;
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
    
//     // Log after Stage 2 for verification
//     LOG_INFO("VIBA Stage2 AFTER:  vel[last]=[{:.3f},{:.3f},{:.3f}], bg=[{:.6f},{:.6f},{:.6f}], ba=[{:.6f},{:.6f},{:.6f}], scale={:.6f}",
//              velocity_params[last_idx][0], velocity_params[last_idx][1], velocity_params[last_idx][2],
//              gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2],
//              accel_bias_params[0], accel_bias_params[1], accel_bias_params[2],
//              scale_params[0]);
    
//     LOG_INFO("VIBA Stage2: iter={}, cost {:.4f} -> {:.4f}", 
//              summary.iterations.size(), summary.initial_cost, summary.final_cost);
    
//     // ==================== Outlier detection ====================
//     int num_inliers = 0;
//     int num_outliers = 0;
    
//     std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
//     std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
//     for (size_t i = 0; i < ba_factors.size(); ++i) {
//         size_t fi = ba_factor_indices[i].first;
//         size_t pi = ba_factor_indices[i].second;
        
//         const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
//         double chi2 = ba_factors[i]->compute_chi_square(params);
        
//         bool is_outlier = (chi2 > 5.991);
//         ba_factors[i]->set_outlier(is_outlier);
        
//         if (is_outlier) {
//             num_outliers++;
//             mp_outlier_count[mappoints[pi]]++;
//         } else {
//             num_inliers++;
//             mp_inlier_count[mappoints[pi]]++;
//         }
//     }
    
//     // Mark bad MapPoints
//     for (const auto& mp : mappoints) {
//         if (mp->IsMarginalized()) continue;
//         int inliers = mp_inlier_count[mp];
//         int outliers = mp_outlier_count[mp];
//         if (inliers == 0 && outliers >= 2) {
//             mp->SetBad();
//         }
//     }
    
//     // ==================== Update results ====================
//     // Update poses (Right Perturbation: T_wb = T_wb_init * exp(delta_xi))
//     // Velocity and bias were already updated in Stage 1
//     for (size_t i = 0; i < frames.size(); ++i) {
//         Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
//         vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
//         vio_360::SE3d T_wb_init_se3(T_wb_inits[i]);
//         vio_360::SE3d T_wb_final = T_wb_init_se3 * delta_T;
//         frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
//     }
    
//     // Update MapPoints
//     for (size_t i = 0; i < mappoints.size(); ++i) {
//         if (!mappoints[i]->IsBad() && !mappoints[i]->IsMarginalized()) {
//             Eigen::Vector3f pos(point_params[i][0], point_params[i][1], point_params[i][2]);
//             mappoints[i]->SetPosition(pos);
//         }
//     }
    
//     result.success = summary.IsSolutionUsable();
//     result.num_inliers = num_inliers;
//     result.num_outliers = num_outliers;
//     result.num_poses_optimized = frames.size();
//     result.num_points_optimized = mappoints.size();
//     result.initial_cost = summary.initial_cost;
//     result.final_cost = summary.final_cost;
//     result.num_iterations = summary.iterations.size();
    
//     return result;
// }

BAResult Optimizer::RunLocalBA(const std::vector<std::shared_ptr<Frame>>& window_frames) {
    BAResult result;
    
    if (window_frames.size() < 2) {
        LOG_WARN("RunLocalBA: need at least 2 frames in window");
        return result;
    }
    
    // Build set of window frame IDs for quick lookup
    std::set<int> window_frame_ids;
    for (const auto& frame : window_frames) {
        window_frame_ids.insert(frame->GetFrameId());
    }
    
    // Step 1: Collect all MapPoints observed by window frames
    std::set<std::shared_ptr<MapPoint>> mappoint_set;
    for (const auto& frame : window_frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->IsValid()) continue;
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                mappoint_set.insert(mp);
            }
        }
    }
    
    std::vector<std::shared_ptr<MapPoint>> mappoints(mappoint_set.begin(), mappoint_set.end());
    
    if (mappoints.empty()) {
        LOG_WARN("RunLocalBA: no valid MapPoints");
        return result;
    }
    
    // Step 2: Only use window frames for BA (ignore out-of-window observations)
    std::vector<std::shared_ptr<Frame>> all_frames(window_frames.begin(), window_frames.end());
    
    // Create frame index map
    std::map<std::shared_ptr<Frame>, size_t> frame_to_idx;
    for (size_t i = 0; i < all_frames.size(); ++i) {
        frame_to_idx[all_frames[i]] = i;
    }
    
    // Parameter blocks for right perturbation
    std::vector<Eigen::Matrix4d> T_wb_inits(all_frames.size());  // Store initial poses
    std::vector<std::array<double, 6>> pose_params(all_frames.size());  // delta_xi (perturbation)
    std::vector<std::array<double, 3>> point_params(mappoints.size());
    
    // Initialize: delta_xi = 0 (right perturbation), store T_wb_init
    for (size_t i = 0; i < all_frames.size(); ++i) {
        T_wb_inits[i] = all_frames[i]->GetTwb().cast<double>();
        pose_params[i] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // delta_xi starts at 0
    }
    
    // Initialize point parameters and create index map
    std::map<std::shared_ptr<MapPoint>, size_t> mp_to_idx;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        Eigen::Vector3f pos = mappoints[i]->GetPosition();
        point_params[i] = {pos.x(), pos.y(), pos.z()};
        mp_to_idx[mappoints[i]] = i;
    }
    
    // Build optimization problem
    ceres::Problem problem;
    std::vector<factor::BAFactor*> factors;
    std::vector<std::pair<size_t, size_t>> factor_indices;  // (frame_idx, mp_idx)
    
    // Add residuals for ALL observations of ALL mappoints
    for (size_t pi = 0; pi < mappoints.size(); ++pi) {
        const auto& mp = mappoints[pi];
        const auto& observations = mp->GetObservations();
        
        for (const auto& obs : observations) {
            auto obs_frame = obs.frame.lock();
            if (!obs_frame) continue;
            
            auto frame_it = frame_to_idx.find(obs_frame);
            if (frame_it == frame_to_idx.end()) continue;
            
            size_t fi = frame_it->second;
            
            // Get the feature from the observation
            const auto& features = obs_frame->GetFeatures();
            if (obs.feature_index < 0 || obs.feature_index >= static_cast<int>(features.size())) continue;
            
            const auto& feature = features[obs.feature_index];
            if (!feature || !feature->IsValid()) continue;
            
            // Skip features near horizontal boundary (ERP wrap-around issue)
            if (IsNearBoundary(feature->GetPixelCoord())) continue;
            
            // Camera parameters
            double cols = static_cast<double>(obs_frame->GetWidth());
            double rows = static_cast<double>(obs_frame->GetHeight());
            factor::CameraParameters cam_params(cols, rows);
            
            Eigen::Matrix4d T_cb = obs_frame->GetTCB().cast<double>();
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() / (m_pixel_noise_std * m_pixel_noise_std);
            
            Eigen::Vector2d obs_pixel(feature->GetPixelCoord().x, feature->GetPixelCoord().y);
            
            // Right perturbation: pass T_wb_init to factor
            auto* cost = new factor::BAFactor(obs_pixel, cam_params, T_cb, T_wb_inits[fi], info);
            factors.push_back(cost);
            factor_indices.push_back({fi, pi});
            
            // Marginalized MapPoints get 2x weight (scale preserving)
            double weight = mp->IsMarginalized() ? 2.0 : 1.0;
            ceres::LossFunction* loss = new ceres::HuberLoss(m_huber_delta);
            if (weight > 1.0) {
                loss = new ceres::ScaledLoss(loss, weight, ceres::TAKE_OWNERSHIP);
            }
            
            problem.AddResidualBlock(
                cost,
                loss,
                pose_params[fi].data(),
                point_params[pi].data()
            );
        }
    }
    
    // Track which pose parameter blocks were actually added to the problem
    std::set<size_t> poses_in_problem;
    for (const auto& [fi, pi] : factor_indices) {
        poses_in_problem.insert(fi);
    }
    
    // For scale stability: fix only the first keyframe
    // All other keyframes are optimized
    constexpr size_t NUM_FIXED_KEYFRAMES = 1;
    int fixed_count = 0;
    int optimized_count = 0;
    
    for (size_t i = 0; i < all_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;
        
        if (i < NUM_FIXED_KEYFRAMES) {
            // Fix oldest N keyframes for scale stability
            problem.SetParameterBlockConstant(pose_params[i].data());
            fixed_count++;
        } else {
            optimized_count++;
        }
    }
    
    // Fix MapPoints that are marked as fixed (initialization points define scale)
    // Optimize MapPoints that are not fixed
    int fixed_mp_count = 0;
    int optimized_mp_count = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (mappoints[i]->IsMarginalized()) {
            problem.SetParameterBlockConstant(point_params[i].data());
            fixed_mp_count++;
        } else {
            optimized_mp_count++;
        }
    }
    
    LOG_DEBUG("  LocalBA: {} window frames (fix oldest {}, optimize {}), {} MapPoints ({} fixed, {} optimize), {} factors",
             window_frames.size(), fixed_count, optimized_count, mappoints.size(), fixed_mp_count, optimized_mp_count, factors.size());
    
    // Solve
    ceres::Solver::Options options = SetupSolverOptions(m_max_iterations);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // Chi-square based outlier detection
    // chi_sq_2D = 5.99146 for 2 DOF with 5% significance level
    constexpr double chi_sq_threshold = 5.99146;
    
    int num_inliers = 0;
    int num_outliers = 0;
    
    std::map<std::shared_ptr<MapPoint>, int> mp_outlier_count;
    std::map<std::shared_ptr<MapPoint>, int> mp_inlier_count;
    
    for (size_t i = 0; i < factors.size(); ++i) {
        size_t fi = factor_indices[i].first;
        size_t pi = factor_indices[i].second;
        
        const double* params[2] = {pose_params[fi].data(), point_params[pi].data()};
        double chi2 = factors[i]->compute_chi_square(params);
        
        bool is_outlier = (chi2 > chi_sq_threshold);
        factors[i]->set_outlier(is_outlier);
        
        if (is_outlier) {
            num_outliers++;
            mp_outlier_count[mappoints[pi]]++;
        } else {
            num_inliers++;
            mp_inlier_count[mappoints[pi]]++;
        }
    }
    
    // Mark MapPoints as bad if ALL observations are outliers
    // But NEVER mark marginalized MapPoints as bad (they preserve scale)
    int bad_mp_count = 0;
    for (const auto& mp : mappoints) {
        if (mp->IsMarginalized()) continue;  // Marginalized MapPoints must not be removed
        int inliers = mp_inlier_count[mp];
        int outliers = mp_outlier_count[mp];
        if (inliers == 0 && outliers >= 2) {
            mp->SetBad();
            bad_mp_count++;
        }
    }
    
    if (bad_mp_count > 0) {
        LOG_DEBUG("  LocalBA: marked {} MapPoints as bad", bad_mp_count);
    }
    
    // Update all optimized frame poses (frames after NUM_FIXED_KEYFRAMES)
    // Right Perturbation: T_wb = T_wb_init * exp(delta_xi)
    for (size_t i = NUM_FIXED_KEYFRAMES; i < all_frames.size(); ++i) {
        if (poses_in_problem.count(i) == 0) continue;  // Skip if not in problem
        
        Eigen::Map<const Eigen::Vector6d> delta_xi(pose_params[i].data());
        vio_360::SE3d delta_T = vio_360::SE3d::exp(delta_xi);
        vio_360::SE3d T_wb_init_se3(T_wb_inits[i]);
        vio_360::SE3d T_wb_final = T_wb_init_se3 * delta_T;
        all_frames[i]->SetTwb(T_wb_final.matrix().cast<float>());
    }
    
    // Update MapPoints that were optimized (not fixed)
    int mp_updated = 0;
    int newly_marginalized = 0;
    for (size_t i = 0; i < mappoints.size(); ++i) {
        if (!mappoints[i]->IsMarginalized()) {
            Eigen::Vector3f new_pos(point_params[i][0], point_params[i][1], point_params[i][2]);
            mappoints[i]->SetPosition(new_pos);
            mp_updated++;
            
            // Increment BA count and marginalize if optimized at least once
            mappoints[i]->IncrementBACount();
            if (mappoints[i]->GetBACount() >= 1) {
                mappoints[i]->SetMarginalized(true);
                newly_marginalized++;
            }
        }
    }
    
    if (newly_marginalized > 0) {
        LOG_DEBUG("  LocalBA: {} MapPoints marginalized (BA count >= 1)", newly_marginalized);
    }
    
    result.success = summary.IsSolutionUsable();
    result.num_inliers = num_inliers;
    result.num_outliers = num_outliers;
    result.num_poses_optimized = window_frames.size() - 1 + fixed_count;  // -1 for fixed first frame
    result.num_points_optimized = mappoints.size();
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();
    
    return result;
}

// ============================================================================
// IMU Initialization Optimization
// ============================================================================

double Optimizer::ComputeScaleMetric(const IMUInitResult& result, double scale) {
    if (!result.success) {
        return std::numeric_limits<double>::max();
    }
    
    // 1. Gravity direction error: how well does gravity point down?
    Eigen::Vector3f g_ideal(0, 0, -1.0f);  // Normalized ideal gravity
    Eigen::Vector3f g_estimated = result.gravity.normalized();
    double gravity_error = (g_estimated - g_ideal).norm();
    
    // 2. Normalize final cost by scale^2 (since residuals scale with translation)
    double normalized_cost = result.final_cost / (scale * scale);
    
    // Combined metric (gravity error is more important)
    double metric = gravity_error * 10.0 + normalized_cost;
    
    return metric;
}

IMUInitResult Optimizer::OptimizeIMUInit(
    const std::vector<std::shared_ptr<Frame>>& frames) {
    
    IMUInitResult result;
    
    // Need at least 3 keyframes for IMU initialization
    if (frames.size() < 3) {
        LOG_WARN("[IMU_INIT] Need at least 3 keyframes, got {}", frames.size());
        return result;
    }
    
    // Check all frames have preintegration (except first)
    for (size_t i = 1; i < frames.size(); ++i) {
        if (!frames[i]->HasIMUPreintegrationFromLastKeyframe()) {
            LOG_WARN("[IMU_INIT] Frame {} missing preintegration", frames[i]->GetFrameId());
            return result;
        }
    }
    
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT] Starting IMU Initialization");
    LOG_INFO("  Keyframes: {}", frames.size());
    LOG_INFO("========================================================");
    
    // Use scale = 1.0 (target_median_depth applied in visual init)
    return OptimizeIMUInitWithScale(frames, 1.0);
}

// ============================================================================
// IMU Initialization with Specific Scale
// ============================================================================

IMUInitResult Optimizer::OptimizeIMUInitWithScale(
    const std::vector<std::shared_ptr<Frame>>& frames,
    double initial_scale) {
    
    IMUInitResult result;
    size_t num_frames = frames.size();
    
    // =========================================================================
    // SETUP: Initialize parameter vectors
    // =========================================================================
    
    // Pose parameters (tangent space): one per frame
    std::vector<std::vector<double>> pose_params(num_frames, std::vector<double>(6, 0.0));
    std::vector<Eigen::Matrix4d> T_wb_inits(num_frames);
    
    // Velocity parameters: one per frame
    std::vector<std::vector<double>> velocity_params(num_frames, std::vector<double>(3, 0.0));
    
    // Bias parameters: shared across all frames
    std::vector<double> gyro_bias_params(3, 0.0);
    std::vector<double> accel_bias_params(3, 0.0);
    
    // Gravity direction: 2D parameterization (theta_x, theta_y)
    std::vector<double> gravity_dir_params(2, 0.0);
    
    // Scale parameter
    std::vector<double> scale_params(1, 1.0);

    // Let's fix scale during IMU initialization
    
    // Initialize poses from frames, applying initial_scale to translations
    for (size_t i = 0; i < num_frames; ++i) {
        T_wb_inits[i] = frames[i]->GetTwb().cast<double>();
        T_wb_inits[i].block<3,1>(0,3) *= initial_scale;
        for (int j = 0; j < 6; ++j) pose_params[i][j] = 0.0;
    }
    
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT] 4-Stage IMU Initialization");
    LOG_INFO("  Keyframes: {}, initial_scale: {:.4f}", num_frames, initial_scale);
    LOG_INFO("========================================================");
    
    // =========================================================================
    // STAGE 0: Gyro Bias Estimation
    // Compare visual rotation with IMU-integrated rotation
    // =========================================================================
    
    LOG_INFO("STAGE 0: Gyro Bias Estimation (Rotation-only)");
    
    {
        // Solve: sum(J_Rg^T * J_Rg) * delta_bg = sum(J_Rg^T * rotation_error)
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        
        int valid_pairs = 0;
        double total_rotation_error_before = 0.0;
        
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint) continue;
            
            double dt = preint->dt_total;
            if (dt < 0.001 || dt > 2.0) continue;
            
            // Visual rotation: R_ij = R_wb_i^T * R_wb_j
            Eigen::Matrix3d R_wb_i = T_wb_inits[i].block<3,3>(0,0);
            Eigen::Matrix3d R_wb_j = T_wb_inits[i+1].block<3,3>(0,0);
            Eigen::Matrix3d R_ij_visual = R_wb_i.transpose() * R_wb_j;
            
            // IMU rotation: delta_R from preintegration
            Eigen::Matrix3d delta_R_imu = preint->delta_R.cast<double>();
            
            // Rotation error: R_err = delta_R_imu^T * R_ij_visual
            Eigen::Matrix3d R_err = delta_R_imu.transpose() * R_ij_visual;
            
            // Log map
            Eigen::AngleAxisd aa(R_err);
            Eigen::Vector3d rotation_error = aa.angle() * aa.axis();
            
            // Jacobian J_Rg from preintegration
            Eigen::Matrix3d J_Rg = preint->J_Rg.cast<double>();
            
            // Accumulate normal equations
            A += J_Rg.transpose() * J_Rg;
            b += J_Rg.transpose() * rotation_error;
            
            total_rotation_error_before += rotation_error.norm() * 180.0 / M_PI;
            valid_pairs++;
        }
        
        if (valid_pairs > 0) {
            // Solve for gyro bias
            Eigen::Vector3d delta_bg = A.ldlt().solve(b);
            gyro_bias_params[0] = delta_bg(0);
            gyro_bias_params[1] = delta_bg(1);
            gyro_bias_params[2] = delta_bg(2);
            
            double avg_err_before = total_rotation_error_before / valid_pairs;
            
            // Verify: recompute rotation error with estimated bias
            double total_rotation_error_after = 0.0;
            for (size_t i = 0; i < num_frames - 1; ++i) {
                auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
                if (!preint || preint->dt_total < 0.001 || preint->dt_total > 2.0) continue;
                
                Eigen::Matrix3d R_wb_i = T_wb_inits[i].block<3,3>(0,0);
                Eigen::Matrix3d R_wb_j = T_wb_inits[i+1].block<3,3>(0,0);
                Eigen::Matrix3d R_ij_visual = R_wb_i.transpose() * R_wb_j;
                
                // Corrected IMU rotation
                Eigen::Matrix3d delta_R_imu = preint->delta_R.cast<double>();
                Eigen::Matrix3d J_Rg = preint->J_Rg.cast<double>();
                Eigen::Vector3d bg_correction = J_Rg * delta_bg;
                Eigen::Matrix3d R_correction = Eigen::Matrix3d::Identity();
                if (bg_correction.norm() > 1e-10) {
                    R_correction = Eigen::AngleAxisd(bg_correction.norm(), 
                        bg_correction.normalized()).toRotationMatrix();
                }
                Eigen::Matrix3d delta_R_corrected = delta_R_imu * R_correction;
                
                Eigen::Matrix3d R_err = delta_R_corrected.transpose() * R_ij_visual;
                Eigen::AngleAxisd aa(R_err);
                total_rotation_error_after += aa.angle() * 180.0 / M_PI;
            }
            double avg_err_after = total_rotation_error_after / valid_pairs;
            
            LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}] rad/s", 
                     delta_bg(0), delta_bg(1), delta_bg(2));
            LOG_INFO("  Rotation error: {:.3f}° -> {:.3f}° ({} pairs)", 
                     avg_err_before, avg_err_after, valid_pairs);
        } else {
            LOG_WARN("  No valid pairs for gyro bias estimation");
        }
    }
    
    // =========================================================================
    // STAGE 1: Initialize Velocity from Visual
    // v_i = (t_{i+1} - t_i) / dt
    // =========================================================================
    
    LOG_INFO("STAGE 1: Velocity Initialization (Visual-based)");
    
    {
        for (size_t i = 0; i < num_frames; ++i) {
            if (i < num_frames - 1) {
                auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
                if (preint && preint->dt_total > 0.001) {
                    Eigen::Vector3d t_curr = T_wb_inits[i].block<3,1>(0,3);
                    Eigen::Vector3d t_next = T_wb_inits[i+1].block<3,1>(0,3);
                    double dt = preint->dt_total;
                    Eigen::Vector3d vel = (t_next - t_curr) / dt;
                    velocity_params[i][0] = vel(0);
                    velocity_params[i][1] = vel(1);
                    velocity_params[i][2] = vel(2);
                }
            } else {
                // Last frame: use previous frame's velocity
                if (num_frames > 1) {
                    velocity_params[i][0] = velocity_params[i-1][0];
                    velocity_params[i][1] = velocity_params[i-1][1];
                    velocity_params[i][2] = velocity_params[i-1][2];
                }
            }
        }
        
        LOG_INFO("  Initialized {} velocities from visual translation", num_frames);
        LOG_DEBUG("  v[0] = [{:.3f}, {:.3f}, {:.3f}]", 
                  velocity_params[0][0], velocity_params[0][1], velocity_params[0][2]);
        LOG_DEBUG("  v[last] = [{:.3f}, {:.3f}, {:.3f}]", 
                  velocity_params[num_frames-1][0], velocity_params[num_frames-1][1], velocity_params[num_frames-1][2]);
    }
    
    // =========================================================================
    // STAGE 2: Gravity Direction Estimation
    // dirG = -sum(R_wb_i * delta_V), then compute R_wg
    // =========================================================================
    
    LOG_INFO("STAGE 2: Gravity Direction Estimation");
    
    {
        Eigen::Vector3d dirG = Eigen::Vector3d::Zero();
        int valid_count = 0;
        
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint || preint->dt_total < 0.001 || preint->dt_total > 2.0) continue;
            
            Eigen::Matrix3d R_wb_i = T_wb_inits[i].block<3,3>(0,0);
            
            // Correct delta_V with estimated gyro bias
            Eigen::Vector3d delta_V = preint->delta_V.cast<double>();
            Eigen::Matrix3d J_Vg = preint->J_Vg.cast<double>();
            Eigen::Vector3d bg(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
            delta_V = delta_V + J_Vg * bg;
            
            // Accumulate: dirG = -sum(R_wb_i * delta_V)
            dirG -= R_wb_i * delta_V;
            valid_count++;
        }
        
        if (valid_count > 0 && dirG.norm() > 1e-6) {
            dirG.normalize();
            
            // gI = [0, 0, -1] (unit gravity in world frame)
            Eigen::Vector3d gI(0, 0, -1);
            
            // Compute rotation from gI to dirG
            Eigen::Vector3d v = gI.cross(dirG);
            double c = gI.dot(dirG);
            
            if (v.norm() > 1e-6) {
                double angle = std::acos(std::max(-1.0, std::min(1.0, c)));
                Eigen::Vector3d axis = v.normalized();
                Eigen::Matrix3d R_wg = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
                
                // Extract gravity_dir_params from R_wg
                Eigen::AngleAxisd aa(R_wg);
                Eigen::Vector3d omega = aa.angle() * aa.axis();
                gravity_dir_params[0] = omega(0);
                gravity_dir_params[1] = omega(1);
                
                LOG_INFO("  Gravity direction: dirG = [{:.3f}, {:.3f}, {:.3f}]",
                         dirG(0), dirG(1), dirG(2));
                LOG_INFO("  Gravity params: [{:.4f}, {:.4f}]", 
                         gravity_dir_params[0], gravity_dir_params[1]);
            } else {
                if (c < 0) {
                    gravity_dir_params[0] = M_PI;
                    gravity_dir_params[1] = 0.0;
                    LOG_INFO("  Gravity is anti-parallel (upside down)");
                } else {
                    gravity_dir_params[0] = 0.0;
                    gravity_dir_params[1] = 0.0;
                    LOG_INFO("  Gravity is already aligned");
                }
            }
        } else {
            LOG_WARN("  Could not estimate gravity direction");
        }
    }
    
    // =========================================================================
    // STAGE 2.5: Estimate Initial Scale from IMU Preintegration
    // Using: p_j - p_i = v_i * dt + 0.5 * g * dt^2 + R_i * delta_p
    // Solve for scale: s = ||R_i * delta_P|| / ||d_visual||
    // delta_P from preintegration is the pure IMU displacement (body frame)
    // =========================================================================
    
    LOG_INFO("STAGE 2.5: Scale Estimation from IMU");
    
    {
        // Compute scale ratio for each segment, then take median
        std::vector<double> scale_ratios;
        
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint || preint->dt_total < 0.001 || preint->dt_total > 2.0) continue;
            
            Eigen::Matrix3d R_wb_i = T_wb_inits[i].block<3,3>(0,0);
            
            // Visual translation
            Eigen::Vector3d t_i = T_wb_inits[i].block<3,1>(0,3);
            Eigen::Vector3d t_j = T_wb_inits[i+1].block<3,1>(0,3);
            Eigen::Vector3d d_visual = t_j - t_i;
            double visual_norm = d_visual.norm();
            
            // Skip if visual displacement is too small (noisy ratio)
            if (visual_norm < 0.01) continue;
            
            // IMU preintegration displacement (corrected with gyro bias)
            Eigen::Vector3d delta_p = preint->delta_P.cast<double>();
            Eigen::Matrix3d J_Pg = preint->J_Pg.cast<double>();
            Eigen::Vector3d bg(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
            delta_p = delta_p + J_Pg * bg;
            
            // Transform to world frame
            Eigen::Vector3d d_imu = R_wb_i * delta_p;
            double imu_norm = d_imu.norm();
            
            // Compute ratio for this segment
            double ratio = imu_norm / visual_norm;
            scale_ratios.push_back(ratio);
        }
        
        if (!scale_ratios.empty()) {
            // Take median for robustness against outliers
            std::sort(scale_ratios.begin(), scale_ratios.end());
            double estimated_scale;
            size_t n = scale_ratios.size();
            if (n % 2 == 0) {
                estimated_scale = (scale_ratios[n/2 - 1] + scale_ratios[n/2]) / 2.0;
            } else {
                estimated_scale = scale_ratios[n/2];
            }
            
            // Clamp to reasonable range
            estimated_scale = std::max(0.1, std::min(10.0, estimated_scale));
            scale_params[0] = estimated_scale;
            LOG_INFO("  Estimated scale: {:.4f} (median of {} ratios, range=[{:.3f}, {:.3f}])", 
                     estimated_scale, n, scale_ratios.front(), scale_ratios.back());
        } else {
            scale_params[0] = 1.0;
            LOG_WARN("  Could not estimate scale, using 1.0");
        }
    }
    
    // =========================================================================
    // STAGE 3: Joint Optimization (Gravity + Accel Bias + Scale + Velocity)
    // Gyro bias is FIXED from Stage 0
    // =========================================================================
    
    LOG_INFO("STAGE 3: Joint Optimization (Gravity + AccelBias + Scale + Velocity)");
    LOG_INFO("  Gyro bias FIXED: [{:.6f}, {:.6f}, {:.6f}]", 
             gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    
    {
        ceres::Problem problem;
        
        // Add InertialGravityScaleFactor between consecutive frames
        int factors_added = 0;
        for (size_t i = 0; i < num_frames - 1; ++i) {
            auto preint = frames[i + 1]->GetIMUPreintegrationFromLastKeyframe();
            if (!preint) continue;
            
            double dt = preint->dt_total;
            if (dt < 0.001 || dt > 2.0) continue;
            
            auto* factor = new factor::InertialGravityScaleFactor(
                preint, ConfigUtils::GetInstance().imu_gravity_magnitude);
            auto* loss = new ceres::HuberLoss(std::sqrt(16.0));
            
            problem.AddResidualBlock(factor, loss,
                pose_params[i].data(),
                velocity_params[i].data(),
                gyro_bias_params.data(),
                accel_bias_params.data(),
                pose_params[i + 1].data(),
                velocity_params[i + 1].data(),
                gravity_dir_params.data(),
                scale_params.data());
            
            factors_added++;
        }
        
        if (factors_added == 0) {
            LOG_WARN("[IMU_INIT] No valid factors added in Stage 3");
            return result;
        }
        
        // Add weak accel bias prior (regularization toward zero)
        Eigen::Vector3d zero_bias = Eigen::Vector3d::Zero();
        double accel_bias_weight = 1.0;
        auto* accel_prior = new factor::BiasPriorFactor(zero_bias, accel_bias_weight);
        problem.AddResidualBlock(accel_prior, nullptr, accel_bias_params.data());
        
        // Add scale prior from Stage 2.5 estimation
        double scale_prior_value = scale_params[0];
        double scale_prior_weight = 0.0045;  // Weak prior - allows optimization to adjust
        auto* scale_prior = new factor::ScalarPriorFactor(scale_prior_value, scale_prior_weight);
        problem.AddResidualBlock(scale_prior, nullptr, scale_params.data());
        
        LOG_INFO("  Scale prior: {:.4f} (weight={:.2f})", scale_prior_value, scale_prior_weight);
        
        // FIXED: All poses, Gyro bias
        for (size_t i = 0; i < num_frames; ++i) {
            problem.SetParameterBlockConstant(pose_params[i].data());
        }
        problem.SetParameterBlockConstant(gyro_bias_params.data());  // Fixed from Stage 0
        
        // FREE: Gravity direction, Accel bias, Scale, Velocities
        // Scale must be positive (lower bound > 0)
        problem.SetParameterLowerBound(scale_params.data(), 0, 0.001);  // Scale > 0.001
        
        // Solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations = 200;
        options.minimizer_progress_to_stdout = false;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        result.initial_cost = summary.initial_cost;
        result.final_cost = summary.final_cost;
        
        LOG_INFO("  Stage 3 result:");
        LOG_INFO("    Scale: {:.4f}", scale_params[0]);
        LOG_INFO("    Gravity params: [{:.4f}, {:.4f}]", gravity_dir_params[0], gravity_dir_params[1]);
        LOG_INFO("    Accel bias: [{:.6f}, {:.6f}, {:.6f}]", 
                 accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
        LOG_INFO("    Cost: {:.6f} -> {:.6f}", summary.initial_cost, summary.final_cost);
    }
    
    // =========================================================================
    // Extract results
    // =========================================================================
    
    double final_scale = initial_scale * scale_params[0];
    
    // Compute gravity vector from direction
    const auto& config = ConfigUtils::GetInstance();
    double gravity_mag = config.imu_gravity_magnitude;
    double theta_x = gravity_dir_params[0];
    double theta_y = gravity_dir_params[1];
    Eigen::Vector3d omega(theta_x, theta_y, 0.0);
    double angle = omega.norm();
    Eigen::Matrix3d R_wg;
    if (angle < 1e-6) {
        R_wg = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = omega / angle;
        R_wg = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }
    Eigen::Vector3d g_I(0, 0, -gravity_mag);
    Eigen::Vector3d gravity = R_wg * g_I;
    
    // Apply final scale to poses (relative to first frame)
    Eigen::Vector3d t0 = T_wb_inits[0].block<3,1>(0,3);
    for (size_t i = 1; i < num_frames; ++i) {
        Eigen::Vector3d t_i = T_wb_inits[i].block<3,1>(0,3);
        Eigen::Vector3d t_i_new = t0 + final_scale * (t_i - t0);
        T_wb_inits[i].block<3,1>(0,3) = t_i_new;
    }
    
    // Apply final scale to MapPoints (relative to first camera frame)
    // Collect all unique MapPoints from frames
    std::set<std::shared_ptr<MapPoint>> unique_mps;
    for (const auto& frame : frames) {
        const auto& features = frame->GetFeatures();
        for (size_t i = 0; i < features.size(); ++i) {
            auto mp = frame->GetMapPoint(static_cast<int>(i));
            if (mp && !mp->IsBad()) {
                unique_mps.insert(mp);
            }
        }
    }
    
    // Scale MapPoints relative to first camera position
    Eigen::Matrix4d Twc_0 = frames[0]->GetTwc().cast<double>();
    Eigen::Matrix4d Tcw_0 = Twc_0.inverse();
    
    for (const auto& mp : unique_mps) {
        Eigen::Vector3d pos_w = mp->GetPosition().cast<double>();
        // Transform to first camera frame
        Eigen::Vector3d pos_c = Tcw_0.block<3,3>(0,0) * pos_w + Tcw_0.block<3,1>(0,3);
        // Apply scale
        pos_c *= final_scale;
        // Transform back to world
        Eigen::Vector3d pos_w_new = Twc_0.block<3,3>(0,0) * pos_c + Twc_0.block<3,1>(0,3);
        mp->SetPosition(pos_w_new.cast<float>());
    }
    
    LOG_INFO("  Scaled {} MapPoints with scale={:.4f}", unique_mps.size(), final_scale);
    
    // Store results
    result.success = true;
    result.gravity = gravity.cast<float>();
    result.Rwg = R_wg.cast<float>();
    result.scale = final_scale;
    result.gyro_bias = Eigen::Vector3f(gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    result.accel_bias = Eigen::Vector3f(accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    
    // Store scaled poses
    result.scaled_poses.resize(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        result.scaled_poses[i] = T_wb_inits[i].cast<float>();
    }
    
    // Store scaled velocities
    result.velocities.resize(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        // Apply scale to velocities
        result.velocities[i] = Eigen::Vector3f(
            velocity_params[i][0] * scale_params[0],
            velocity_params[i][1] * scale_params[0],
            velocity_params[i][2] * scale_params[0]);
    }
    
    LOG_INFO("========================================================");
    LOG_INFO("[IMU_INIT] Initialization Complete");
    LOG_INFO("  Final scale: {:.4f}", final_scale);
    LOG_INFO("  Gravity: [{:.3f}, {:.3f}, {:.3f}] (norm={:.3f})",
             gravity(0), gravity(1), gravity(2), gravity.norm());
    LOG_INFO("  Gyro bias: [{:.6f}, {:.6f}, {:.6f}]",
             gyro_bias_params[0], gyro_bias_params[1], gyro_bias_params[2]);
    LOG_INFO("  Accel bias: [{:.6f}, {:.6f}, {:.6f}]",
             accel_bias_params[0], accel_bias_params[1], accel_bias_params[2]);
    LOG_INFO("========================================================");
    
    return result;
}

} // namespace vio_360
