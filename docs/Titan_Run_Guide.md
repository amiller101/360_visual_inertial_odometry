# Titan 360-VIO Run Guide

This guide describes how the 360-VIO program runs and gives suggestions for processing panoramic video  
from an Insta360 Titan in row-crop environments.

See Section 6 for the run command.

---

## 1  Dataset layout

Each sequence directory must contain:

```
<sequence>/
  images/              stitched ERP frames (.jpg, .jpeg, or .png)
  cam_timestamps.txt   one frame timestamp per line (seconds, monotonically increasing)
  imu_data.csv         header + rows: timestamp,ax,ay,az,gx,gy,gz
```

- Images are loaded in **lexicographic filename order**; use zero-padded names
(e.g. `000001.png`)
- The app truncates to the **shorter** of image count and timestamp count.
- IMU units: accelerometer in **m/s²**, gyroscope in **rad/s**.

You may validate the dataset before running:

```bash
python3 scripts/validate_titan_dataset.py /path/to/sequence
```

---

## 2  Calibration — T_BC

The estimator needs `extrinsics.T_BC`, the 4×4 rigid transform from the camera
frame to the body (IMU) frame:  `p_body = T_BC · p_camera`.

`config/default_config.yaml` ships with T_BC = identity.  This is a safe  
starting point if you have no calibration data yet — it assumes the camera and  
IMU share the same orientation.  The estimator will still run, but scale  
and IMU initialization will be degraded unless the assumption roughly  
holds.

`**config/titan_config.yaml**` uses a T_BC produced by sequence estimation
from a recorded Titan run (rotation only; translation set to zero).  Use this
as a baseline and replace it with a Kalibr result if possible.

See `docs/Titan_Calibration_Suggestions.md` for options on producing a better
T_BC.

---

## 3  Startup motion recommendation

Row-crop motion is low-excitation.  Monocular initialisation and IMU scale
estimation both depend on **early parallax and rotation**:

- If possible, begin each capture with 10–20 second**s** of deliberate pitch, roll, and yaw
motion before entering any straight row.
- Alternatively, begin with the the highest excitement possible, such as a uturn in otherwise straightline movement. 
- Starting with pure forward motion generally causes initialisation to fail.

---

## 4  Image resolution and downscaling

The Titan outputs stitched ERP frames at **3840 × 1920** (native).  `camera.width`
and `camera.height` in the config set the **working resolution** the app uses
internally — `main.cpp` rescales every loaded frame to that size with
`cv::INTER_AREA` before processing.

`titan_config.yaml` sets `960 × 480`, a **4× downscale**.  This is not an FOV
setting — the full 360° ERP content is preserved, just at lower resolution.

All other pixel-unit parameters in the config (`boundary_margin`,
`min_distance`, `optical_flow.window_size`, `low_motion_max_flow_px`, etc.) are
in units of the **working resolution**, not native.

To process at native or half resolution, change `width`/`height` and re-tune
the pixel-unit parameters proportionally.

---

## 5  Three-stage initialisation

Before useful trajectory output is available the estimator runs three stages:


| Stage | State name        | What happens                                                                                                                                                                                                                      |
| ----- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `NOT_INITIALIZED` | Features are detected and tracked; the initializer accumulates a window of frames and attempts a 5-point monocular reconstruction. **Camera only, no IMU.** Succeeds when enough parallax and inliers are found.                  |
| 2     | `VISUAL_ONLY`     | A visual map exists; PnP tracking runs each frame. Keyframes are created at a fixed 0.25 s cadence. IMU data is **collected and preintegrated** but not yet fused. After ≥10 keyframes and ≥2 s, IMU initialisation is attempted. |
| 3     | `VIO`             | IMU scale, gravity, accelerometer bias, and velocities have been solved. Full visual-inertial odometry runs. Keyframe creation switches to parallax + time-fallback logic.                                                        |


**"Monocular initialisation"** (Stage 1) means the initial map bootstrap uses  
**camera geometry only.** IMU data is being buffered during this stage but is not used until Stage 2→3.

The viewer pauses when Stage 1 → 2 succeeds so you can confirm initialisation before continuing.

---

## 6  Running the estimator

Pass the config path explicitly.  `LIBGL_ALWAYS_SOFTWARE=1` prevents
OpenGL crashes on headless systems:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./build/bin/360_vio_example \
  /path/to/sequence \
  /path/to/config/titan_config.yaml
```

The viewer starts **paused**; press `Space` (or the Pause button) to begin
processing.  When monocular + IMU initialisation succeeds the viewer pauses
again — press `Space` to continue into full VIO.

### Trajectory autosave

The estimator writes `estimated_trajectory_autosave.txt` inside the sequence
directory every `video_output.trajectory_autosave_interval_frames` frames
(default 200, roughly every ~7 s at 30 fps).  This prevents data loss if the
process is killed before the viewer's **Finish & Save** button is pressed.

The autosave file contains only keyframes processed so far and is overwritten
on each interval.  The final `estimated_trajectory.txt` written by Finish & Save
is the complete, authoritative output.

---

## 7  Keyframe behaviour on straight segments

### Problem

After IMU initialisation the estimator creates keyframes based on **median pixel
parallax** versus the last keyframe.  Along a straight crop row, parallax stays
low even at normal driving speed, so keyframe creation can stall — leaving the
sliding-window map without new constraints and allowing bias integration to
accumulate.

### Solution (implemented in this fork)

Two complementary mechanisms are active in `titan_config.yaml`:


| Mechanism                | Config key                                      | Effect                                                                   |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------ |
| Lower parallax threshold | `tracking.min_parallax_for_keyframe: 6.0`       | Fires keyframes on moderate parallax typical of slow row motion          |
| Time-based fallback      | `tracking.vio_time_fallback_keyframe_sec: 0.65` | Forces a keyframe every 0.65 s when PnP succeeds, regardless of parallax |


The time fallback only activates in **VIO state** — it will not interfere with
initialisation.

**Tuning the time fallback:**

- Lower (0.45–0.55 s) → denser keyframes, more constraints on straight rows.
- Higher (0.9–1.2 s) → fewer keyframes, lighter map.
- `0.0` disables the fallback entirely (parallax-only, original behaviour).

---

## 8  Feature masking (ERP-specific)

Features are suppressed in four border regions controlled by
`camera.boundary_margin` (default 20 px at 960-wide):

- **Left / right**: ERP wrap-around seam — features here split across the
discontinuity and produce spurious flow.
- **Top / bottom**: polar regions where sensor coverage is weakest and fisheye
  distortion is most severe, producing geometrically unreliable features.

If corner features are still detected on the physical rig body, raise
`camera.boundary_margin` to 35–60 px.

---

## 9  Persistent low-motion filter

Tracks that survive rotation RANSAC but move less than `low_motion_max_flow_px`
pixels for `low_motion_min_age` consecutive frames are discarded.  These are
typically rig-fixed points (stitch seams, rig body) that produce zero parallax
and degrade scale and structure estimates.

```yaml
tracking:
  filter_persistent_low_motion: 1
  low_motion_min_age: 15        # frames
  low_motion_max_flow_px: 0.18  # pixels
```

The filter is **automatically disabled** by the estimator during monocular and
IMU initialisation so it cannot reject the sparse flow that initialisation
depends on.  It is enabled once the system transitions to VIO state.

---

## 10  IMU init crash fix

Ceres `SetParameterBlockConstant` previously crashed when called on parameter
blocks that were not added to the Stage-3 problem (disconnected frames with no
IMU factors).  The fix checks `problem.HasParameterBlock()` before each call and
logs a warning for any skipped blocks.

---

## 11  Tuning order

Adjust one group at a time and record results in `docs/titan_benchmark_template.csv`:

1. `initialization.feature_detection.`* — feature spread and count for the
  startup motion window.
2. `initialization.ransac_threshold`, `min_features`, `max_reprojection_error` —
  controls how strict the monocular initialization is.
3. `tracking.min_parallax_for_keyframe`, `vio_time_fallback_keyframe_sec`,
  `window_size` — keyframe density and feature freshness on rows.
4. `tracking.filter_persistent_low_motion`, `low_motion_min_age`,
  `low_motion_max_flow_px` — rig-track suppression.
5. `camera.boundary_margin` — ERP exclusion zone.
6. `imu.`* noise values — May want to calibrate if the estimator
  has consistent scale or drift issues.

---

## 12  Output files


| File                                | Written by           | Contents                                              |
| ----------------------------------- | -------------------- | ----------------------------------------------------- |
| `estimated_trajectory.txt`          | viewer Finish & Save | All keyframes in TUM format: `t tx ty tz qx qy qz qw` |
| `estimated_trajectory_autosave.txt` | periodic mid-run     | Same format, partial, overwritten each interval       |


Both files cover all **marginalized keyframes** (left the sliding window) plus
the current window at save time.  The live Pangolin overlay shows only the
current window; the saved files show the full trajectory.

---

## 13  Post-processing

The trajectory saved by this program (`estimated_trajectory.txt`, TUM format:
`timestamp tx ty tz qx qy qz qw`) is the starting point for downstream
processing such as GPS alignment and NeRF training.  Those workflows are handled
by tools in a separate repository; see that project's documentation for details.