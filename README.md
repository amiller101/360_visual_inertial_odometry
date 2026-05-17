# 360° Visual-Inertial Odometry — Titan Fork

This is a fork of [93won/360_visual_inertial_odometry](https://github.com/93won/360_visual_inertial_odometry), extended for use with stitched equirectangular video from an **Insta360 Titan**
in agricultural row-crop environments.

For a full description of every change made to the original codebase, see
`[docs/Changes_Implemented.md](docs/Changes_Implemented.md)`.

---

## Dependencies

- **C++17** or later
- **CMake** >= 3.14
- **OpenCV** >= 4.0
- **Eigen3** >= 3.3
- **Ceres Solver** >= 2.0 (included as submodule)
- **Pangolin** (included as submodule)
- **spdlog** (included as submodule)

## Build

```bash
git clone --recursive <your-fork-url>
cd 360_visual_inertial_odometry
./build.sh
```

The executable is generated at `build/bin/360_vio_example`.

---

## Running

Pass the sequence directory and a config file explicitly:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./build/bin/360_vio_example \
  /path/to/sequence \
  /path/to/config/titan_config.yaml
```

`LIBGL_ALWAYS_SOFTWARE=1` prevents OpenGL crashes on headless or
software-render systems.

See `[docs/Titan_Run_Guide.md](docs/Titan_Run_Guide.md)` for the full workflow,
including dataset layout, calibration, tuning, and output files.

---

## Dataset

### Original benchmark

The original dataset is from the **360VIO** benchmark:

📄 [360VIO: Robust Visual-Inertial Odometry Using 360 Degree Camera](https://ieee-dataport.org/documents/360vio-robust-visual-inertial-odometry-using-360-degree-camera)

🔗 [Pre-parsed dataset (Google Drive)](https://drive.google.com/drive/folders/1Cry1wAP2cYRwB4armNdCuuae5ZyYmsxm?usp=drive_link)

### Titan sequences

Expected layout for Titan data:

```
<sequence>/
  images/             stitched ERP frames (.jpg, .jpeg, or .png)
  cam_timestamps.txt  one timestamp per line (seconds, monotonically increasing)
  imu_data.csv        header + rows: timestamp,ax,ay,az,gx,gy,gz
```

Validate a sequence before running:

```bash
python3 scripts/validate_titan_dataset.py /path/to/sequence
```

---

## Documentation


| File                                                                             | Description                                          |
| -------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `[docs/Titan_Run_Guide.md](docs/Titan_Run_Guide.md)`                             | Full operator guide for Titan data                   |
| `[docs/Titan_Calibration_Suggestions.md](docs/Titan_Calibration_Suggestions.md)` | T_BC calibration options and record template         |
| `[docs/Changes_Implemented.md](docs/Changes_Implemented.md)`                     | All changes made relative to the upstream repository |


---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.