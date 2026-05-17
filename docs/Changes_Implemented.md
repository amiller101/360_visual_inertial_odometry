# Changes Implemented (Titan Fork)

This document records every change made to the original [93won/360_visual_inertial_odometry](https://github.com/93won/360_visual_inertial_odometry) codebase in this fork. 
Changes are motivated by use with stitched equirectangular
video from an Insta360 Titan in agricultural row-crop environments.

---

## Bug fixes

### Ceres crash during IMU initialisation — `src/optimization/Optimizer.cpp`

**Problem:** In Stage 3 of `OptimizeIMUInitWithScale`, the code called
`problem.SetParameterBlockConstant()` unconditionally for every pose parameter
block.  When a frame had no IMU factor connecting it to the problem (e.g. a
disconnected frame at the edge of the window), Ceres aborted with a fatal error
because the block was never added to the problem.

**Fix:** Wrapped each call with `problem.HasParameterBlock()` and added a warning
log for any skipped blocks.

```cpp
// Before
problem.SetParameterBlockConstant(pose_params[i].data());

// After
if (problem.HasParameterBlock(pose_params[i].data())) {
    problem.SetParameterBlockConstant(pose_params[i].data());
}
```

---

## Feature tracker improvements — `src/processing/FeatureTracker.cpp/.h`

### 1. Top and bottom boundary masking

**Problem:** The original boundary mask only excluded the left and right edges of
the ERP frame (wrap-around seam).  On multi-sensor ERP output such as the Titan,
the top and bottom rows correspond to the polar regions of the sphere where the
sensor coverage is weakest and fisheye distortion is most severe.  Features
detected there are geometrically unreliable and produce noisy optical flow that
contaminates rotation RANSAC and parallax measurements.

**Fix:** Extended the boundary mask to also zero out `boundary_margin` pixels
along the top and bottom of the frame.  Controlled by the existing
`camera.boundary_margin` config key.

### 2. Persistent low-motion track suppression

**Problem:** On ERP imagery, features on the physical rig or stitch seams can
survive rotation RANSAC (they move consistently with the camera rotation) but
have zero translational flow.  They accumulate as high-age tracks that drag the
median parallax toward zero, preventing keyframe creation and degrading scale
estimates.

**Fix:** After RANSAC inlier selection, any track that is older than
`low_motion_min_age` frames **and** has optical-flow magnitude below
`low_motion_max_flow_px` pixels is discarded.  The filter is controlled by three
new config keys under `tracking`:

| Key | Default | Description |
|---|---|---|
| `filter_persistent_low_motion` | `0` | Enable (1) or disable (0) the filter |
| `low_motion_min_age` | `12` | Minimum track age in frames before suppression |
| `low_motion_max_flow_px` | `0.25` | Flow threshold below which a track is suppressed |

**Important:** the filter is automatically **disabled during monocular and IMU
initialisation** and **enabled only in VIO state**.  This prevents it from
discarding the sparse flow that the initialisation pipeline depends on.  The
`Estimator` calls `SetPersistentLowMotionFilterEnabled(bool)` on each frame to
manage this.

---

## Estimator improvements — `src/processing/Estimator.cpp`

### 1. Time-based keyframe fallback in VIO state

**Problem:** After IMU initialisation, keyframes were created only when median
pixel parallax exceeded a threshold.  Along a straight crop row, parallax stays
low even when the robot is moving, so keyframe creation stalls.  Without new
keyframes the sliding-window BA cannot constrain the map and IMU bias integration
accumulates unchecked.

**Fix:** Added a secondary keyframe trigger: if no parallax-based keyframe has
been created for `vio_time_fallback_keyframe_sec` seconds **and** PnP tracking
succeeded, a keyframe is forced.  The fallback only activates in `VIO` state and
is independent of the fixed 0.25 s cadence used during `VISUAL_ONLY`.

```yaml
# titan_config.yaml
tracking:
  vio_time_fallback_keyframe_sec: 0.65  # 0.0 = disabled (original behaviour)
```

### 2. Low-motion filter gated to VIO state

The call to `SetPersistentLowMotionFilterEnabled` (described above) is placed in
`Estimator::TrackFeatures()` so that the filter state tracks the estimator state
machine automatically without any manual intervention.

---

## Runtime resilience — `app/main.cpp`

### Periodic trajectory autosave

**Problem:** On long runs or when the process is killed by an OOM event before
the viewer's "Finish & Save" button is pressed, all trajectory data is lost.

**Fix:** Added a mid-run autosave that writes `estimated_trajectory_autosave.txt`
in the sequence directory every `trajectory_autosave_interval_frames` frames.
The file is overwritten on each interval; it contains all marginalized keyframes
plus the current sliding window at save time.  The final authoritative save via
the viewer is unchanged.

```yaml
video_output:
  trajectory_autosave_interval_frames: 200  # 0 = disabled
```

---

## Configuration system — `src/util/ConfigUtils.h/.cpp`

Four new fields added to `ConfigUtils` to support the above features:

| Field | Type | Default | Config key |
|---|---|---|---|
| `tracking_filter_persistent_low_motion` | `bool` | `false` | `tracking.filter_persistent_low_motion` |
| `tracking_low_motion_min_age` | `int` | `12` | `tracking.low_motion_min_age` |
| `tracking_low_motion_max_flow_px` | `float` | `0.25` | `tracking.low_motion_max_flow_px` |
| `tracking_vio_time_fallback_keyframe_sec` | `float` | `0.0` | `tracking.vio_time_fallback_keyframe_sec` |
| `trajectory_autosave_interval_frames` | `int` | `200` | `video_output.trajectory_autosave_interval_frames` |

All new keys are optional in the YAML; if absent the defaults above apply so
existing config files continue to work without modification.

---

## Config files

### `config/titan_config.yaml` (new)

A complete config profile for the Insta360 Titan with stitched ERP output at
960×480 working resolution (4× downscale from the native 3840×1920).  Key
differences from the default:

| Parameter | Default | Titan | Reason |
|---|---|---|---|
| `tracking.min_parallax_for_keyframe` | 20.0 | 6.0 | Crop-row motion produces low parallax |
| `tracking.vio_time_fallback_keyframe_sec` | 0.0 | 0.65 | Ensures keyframes on straight rows |
| `tracking.filter_persistent_low_motion` | 0 | 1 | Removes rig/seam fixed tracks |
| `tracking.low_motion_min_age` | 12 | 15 | Tuned for Titan track lifetime |
| `tracking.low_motion_max_flow_px` | 0.25 | 0.18 | Tighter threshold for ERP at this resolution |
| `camera.boundary_margin` | 20 | 20 | Masks ERP seam and polar distortion zone |
| `initialization.min_parallax` | 20.0 | 12.0 | Looser for low-excitation startup |
| `initialization.min_features` | 1000 | 20 | Looser for agricultural scenes |
| `initialization.max_reprojection_error` | 5.0 | 8.0 | Relaxed for ERP reprojection units |
| `video_output.trajectory_autosave_interval_frames` | 0 | 200 | Enabled for long field runs |

### `config/default_config.yaml` (modified)

- T_BC reset to identity (the original shipped a specific calibrated value that
  is not appropriate as a generic default).
- New keys added with safe off/zero defaults so the file stays a valid
  standalone config.
- `gravity_magnitude` added to `imu` block (was missing).
- `video_output` block added.

---

## Documentation added

| File | Contents |
|---|---|
| `docs/Titan_Run_Guide.md` | Full operator guide: dataset layout, calibration, resolution/downscaling, three-stage initialisation, keyframe behaviour, masking, tuning, output files |
| `docs/Titan_Calibration_Suggestions.md` | T_BC options (identity / sequence estimation / Kalibr), calibration record template, validation checklist |
| `docs/Changes_Implemented.md` | This file |
| `config/titan_config.yaml` | Titan-specific config profile (see above) |
