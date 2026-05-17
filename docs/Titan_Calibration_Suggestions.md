# Titan Calibration Suggestions

This document covers how to produce a `T_BC` for the Titan rig and what to
record when you do.  See `docs/Titan_Run_Guide.md` § 2 for where `T_BC` is used
and what ships in each config file by default.

---

## What T_BC is

`T_BC` is the 4×4 rigid transform from the **camera** frame to the **body (IMU)**
frame:

```
p_body = T_BC · p_camera
```

It lives under `extrinsics.T_BC` in the config file.  The rotation part
describes how the camera axes relate to the IMU axes; the translation part
describes the physical offset between them (often set to zero when the devices
are close together).

---

## Option 1 — Identity (no calibration)

Set T_BC to identity.  The estimator will run but IMU initialisation and scale
recovery will be degraded unless the camera and IMU happen to share a similar
orientation.  Use only for a quick smoke-test.

`config/default_config.yaml` ships with T_BC = identity.

---

## Option 2 — Sequence Estimation (rotation only, currently in use)

Use an existing rover recording to estimate the rotation component from
visual and IMU relative rotations.  The value currently in
`config/titan_config.yaml` was produced this way.  Translation remains zero.
This is usually much better than identity for heading and scale stability.

---

## Option 3 — Full calibration (Kalibr / TartanCalib, requires AprilTags)

1. Record a calibration sequence with high 6-DOF excitation (yaw, pitch, roll,
  translation) and a calibration board visible in the stitched ERP output.
2. Run your calibration toolchain.
3. If the tool outputs `T_cam_imu` (IMU → camera), invert it:
  `T_BC = inverse(T_cam_imu)`.
4. Paste the result into your config under `extrinsics.T_BC`.

---

## Suggested Calibration Records

Keep a note of the following whenever you update T_BC:


| Field                             | Value                                      |
| --------------------------------- | ------------------------------------------ |
| Method                            | e.g. Kalibr, sequence estimation, identity |
| ERP resolution during calibration | e.g. 3840×1920                             |
| T_BC matrix                       | (paste)                                    |
| Quality metric                    | e.g. reprojection error, residual angle    |
| Timestamp source                  | e.g. hardware sync, software stamp         |
| Any clock offset applied          | seconds                                    |


---

## Validating a new T_BC

- The rotation block should be orthonormal (determinant ≈ +1, columns ≈ unit
length).
- Translation should be physically plausible: centimetre-scale, not metres.
- On a straight-line test run, the estimated trajectory length should match
rough ground-truth (GPS or measured distance).
- `scripts/validate_titan_dataset.py` confirms the IMU data has no zero-sample
intervals that would silently starve the IMU integrator.

---

## Time synchronisation

The `T_BC` estimator assumes camera and IMU timestamps are already in the **same time**  
**base** with no offset.