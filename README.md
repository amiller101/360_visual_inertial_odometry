# 360° Visual-Inertial Odometry

A real-time Visual-Inertial Odometry (VIO) system using 360° equirectangular cameras with IMU integration.

![360 VIO Demo](docs/demo.gif)


## Dependencies

- **C++17** or later
- **CMake** >= 3.14
- **OpenCV** >= 4.0
- **Eigen3** >= 3.3
- **Ceres Solver** >= 2.0 (included as submodule)
- **Pangolin** (included as submodule)
- **spdlog** (included as submodule)

## Build Instructions

### 1. Clone the repository

```bash
git clone --recursive https://github.com/93won/360_visual_inertial_odometry.git
cd 360_visual_inertial_odometry
```

### 2. Build the project

```bash
./build.sh
```


The executable will be generated at `build/bin/360_vio_example`.

## Dataset

### Original Dataset Source

The dataset is from the **360VIO** benchmark:

📄 **Paper**: [360VIO: Robust Visual-Inertial Odometry Using 360 Degree Camera](https://ieee-dataport.org/documents/360vio-robust-visual-inertial-odometry-using-360-degree-camera)

### Pre-parsed Dataset (Recommended)

For convenience, we provide pre-parsed datasets ready to use:

🔗 **Google Drive**: [Download Dataset](https://drive.google.com/drive/folders/1Cry1wAP2cYRwB4armNdCuuae5ZyYmsxm?usp=drive_link)

## Running the Demo

```bash
cd build
./bin/360_vio_example /path/to/dataset/seq1_vio/
```

### Example

```bash
./bin/360_vio_example /home/user/data/360/seq1_vio/
```



## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.


