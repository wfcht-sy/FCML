# WARA-PX4 Installation Guide

![PX4](https://img.shields.io/badge/PX4-v1.13.3-blue)

### Dependencies and Prerequisites

We developed and tested WARA-PX4 on **Ubuntu 20.04 LTS** and **Python 3.10**.
You need to build PX4 in order to use simulators.
Before building the PX4, you must first install the **Developer Toolchain** for your host operating system and target hardware.

The following instructions use a bash script set up the PX4 development environment on Ubuntu 20.04.
This instruction is intended for **simulation and NuttX (Pixhawk) targets**.
For other target platform please refer to [https://docs.px4.io/v1.13/en/dev_setup/dev_env_linux_ubuntu.html](https://docs.px4.io/v1.13/en/dev_setup/dev_env_linux_ubuntu.html).

**IMPORTANT NOTE**
1. We recommend to **create a virtual environment** before proceeding the installation.
2. The script is intended to be run on *clean* Ubuntu LTS installations, and may not work if run "on top" of an existing system, or on a different Ubuntu release.

### One-line Quick Setup

We provide a bash script to setup the PX4-Autopilot and its dependencies:
```bash
   bash Install_WARA-PX4.sh
```
Following the instruction to download and setup WARA-PX4.
Reboot the computer to complete the setup.
After that, read **First Build** section to verify the installation.

### Manual Setup Guide
If the auto installation was interrupted, or a manual configuration is preferred,
read the following manual setup guide to install the PX4-Autopilot and its toolchain:
1. Retrieve the vanilla PX4-Autopilot
   ```bash
   # Replace 'WARA-PX4' with your desired PX4 folder name
   git clone --depth 1 --branch v1.13.3 --recursive https://github.com/PX4/PX4-Autopilot.git WARA-PX4
   cd WARA-PX4
   ```
   If any submodules failed to download, run `git submodule update --init --recursive` in the PX4 folder
2. Apply the patch to integrate WARA.
   Please ensure the patch file (`WARA-PX4_v1.13.3.patch`) is in the directory **above** the PX4 folder.
   ```bash
   git apply -C1 --ignore-whitespace < ../WARA-PX4_v1.13.3.patch
   ```

3. Install the required Ubuntu toolchain and simulation dependencies:
   ```bash
   bash Tools/setup/ubuntu.sh
   ```
4. **[IMPORTANT] Python Environment Fix**

   PX4 1.13.3 is incompatible with empy 4.x+. Please rollback to legacy version:
    ```bash
    pip uninstall empy
    pip install empy==3.3.4
    ```
5. Restart the computer after the toolchain is installed.
6. Verify the installation with `make px4_sitl_wavelet gazebo`

### First Build (with Gazebo-Classic simulator)
1. Download [**QGroundControl**](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html) for drone control.
2. Navigate to the autopilot folder and build PX4 in Gazebo-Classic simulation with `make px4_sitl_wavelet gazebo`.
   The `_wavelet` suffix specify the build target that include WARA implementation.
3. Once the build is completed, the command terminal will show a "PX4" text figure, and the Gazebo-Classic window will pop up.
4. Run QGroundControl, wait until it connected to the drone.
5. Following the [**official guide**](https://docs.qgroundcontrol.com/Stable_V4.3/en/qgc-user-guide/setup_view/parameters.html) to change flight parameters.
   Load the `.params` file provided in the `../parameters` folder. Configure other parameters as needed.
6. In the command terminal, press Ctrl-C to stop the PX4 and simulation.
7. Run `make px4_sitl_wavelet gazebo` again to apply the changed parameters.
8. The WARA-PX4 is now ready for simulation experiments. Following [**this QGC guide**](https://docs.qgroundcontrol.com/Stable_V4.3/en/qgc-user-guide/fly_view/fly_view.html) to command the vehicle.

## Directory Structure

```
WARA-PX4
├── src
│   │
│   ├── lib
│   │   ├── wavelet  <--- THE CORE OF WARA
│   │   │   Codes for the Online Wavelet Packet Decomposition
│   │   │   Related to the Section III.B to III.D in the original paper
│   │   │
│   │   ├── drivers  <--- middleware for IMU sensors
│   │   │   ├── accelerometer
│   │   │   └── gyroscope
│   │   │       The call site of simulated attacks, detectors and the recovery.
│   │   │
│   │   ├── fault_detector
│   │   │   Implement the CuSum and EMA detection algorithms.
│   │   │
│   │   └── sensor_attack
│   │       Simulated IMU attacks
│   │
│   └── modules
│       ├── ekf2
│       │   Revised EKF2Selector to implement the fault isolation (Section III.E)
│       │
│       ├── sensors
│       │   Add simulated attack parameters here.
│       │   
│       └── wavelet_denoiser
│           Hosts the parameters for wavelet-based recovery.
│           Also used for testing the runtime overhead.
│
│
├── boards  <--- compile configs (and the command to compile them)
│   │
│   ├── emlid/navio2
│   │   ├── wavelet.px4board  (make emlid_navio2_wavelet)
│   │   │   Raspberry Pi (32-Bits), for official Navio2 Image.
│   │   │
│   │   └── wavelet-aarch64.px4board  (make emlid_navio2_wavelet-aarch64)
│   │       Raspberry Pi (64-Bits), for HITL evaluation with UnRocker
│   │       Note: EXPERIMENTAL, use at own risk.
│   │
│   └── px4
│       ├── fmu-v3/wavelet.px4board  (make px4_fmu-v3_wavelet)
│       │   Pixhawk 2.4.8 (STM32F4, 256 kB Mem, 2 MB Flash)
│       │   Not recommended unless reduce WPT nodes.
│       │
│       ├── fmu-v5/wavelet.px4board  (make px4_fmu-v5_wavelet)
│       │   CUAV V5+ (STM32F7, 512 kB Mem, 2 MB Flash)
│       │   Evaluated with real-world attack test.
│       │
│       ├── fmu-v6c/wavelet.px4board  (make px4_fmu-v6c_wavelet)
│       │   Pixhawk 6C Mini (STM32H7, 1 MB Mem, 2MB Flash)
│       │   Evaluated with real-world attack test.
│       │
│       └── sitl/wavelet.px4board  (make px4_sitl_wavelet)
│           For SITL and HITL evaluations
│
│
├── msgs  <--- uORB message definitions (List modified .msg files only)
│   ├── sensor_accel.msg 
│   ├── sensor_accel_fifo.msg
│   ├── sensor_gyro.msg 
│   ├── sensor_gyro_fifo.msg
│   │   The raw, recovered, and the groundtruth (in HITL) measurements
│   │
│   ├── sensor_accel_errors.msg
│   ├── sensor_gyro_errors.msg
│   │   Record residuals and detection status for fault isolation
│   │
│   ├── wavelet_coefficient_accel.msg
│   ├── wavelet_coefficient_gyro.msg
│   │   Record wavelet coefficients in flight
│   │
│   ├── wavelet_correction_accel.msg
│   ├── wavelet_correction_gyro.msg
│   │   Record estimated noise & interference in flight
│   │
│   └── wavelet_status.msg
│       Record device_id and wavelet packets decomposition paths
│
│
└── ROMFS\px4fmu_common <--- [DEPRECATED] Auto-init wavelet_denoiser module for overhead test
    ├── init.d\rcS
    └── init.d-posix\rcS

```