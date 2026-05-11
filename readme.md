# DTW-Triplet: Offline Training & Online Adaptive Control Framework

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Platform](https://img.shields.io/badge/Platform-Ubuntu_20.04%20%7C%2022.04-orange)
![PX4](https://img.shields.io/badge/PX4-Autopilot-green)

This repository contains the complete codebase for offline model training, online adaptive flight control evaluation, and experiment result visualization. The core objective is to validate tracking and wind-rejection performance of various control strategies (Baseline PID, INDI, L1, Neural-Fly, and our proposed FCML method) under simulated wind disturbances.

### Dependencies and Prerequisites

We developed and tested DTW-Triplet on **Ubuntu 20.04/22.04 LTS** and **Python 3.9**.
You need to build PX4 in order to use simulators for online flight evaluations.
Before building the PX4, you must first install the **Developer Toolchain** for your host operating system and target hardware.

**Git LFS** is required for downloading large data files (model checkpoints, datasets).
- Dependencies Disk Space: ~7.7 GB
- Repository with datasets: ~2.9 GB

Install Git LFS on Ubuntu:
```bash
sudo apt install git-lfs
git lfs install
```

**IMPORTANT NOTE**
1. We recommend to **create a virtual environment** before proceeding the installation.
2. The installation scripts are intended to be run on *clean* Ubuntu LTS installations.

### One-line Quick Setup

We provide a bash script to setup the project dependencies, conda environment, and PX4-Autopilot toolchains.

First, retrieve the repository and pull all LFS files:
```bash
git clone https://github.com/wfcht-sy/FCML.git
cd FCML
git lfs pull
```

Then, run the one-line setup script:
```bash
bash setup.sh all
```
Following the instruction to download and setup the environment.
If you plan to run online flight evaluations in Gazebo, also run:
```bash
bash setup.sh px4
```
Reboot the computer to complete the setup.
After that, read the **Quick Start** section to verify the installation.

### Manual Setup Guide
If the auto installation was interrupted, or a manual configuration is preferred, read the following manual setup guide:

1. **Create the Conda Environment**
   Install Miniconda if you don't have it, then:
   ```bash
   conda env create -f environment.yml
   conda activate fcml
   ```

2. **Install PyTorch**
   The `environment.yml` intentionally **does not include PyTorch**. Install it manually based on your system:
   *(CPU only)*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   *(GPU CUDA 12.4)*
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install Additional Dependencies**
   ```bash
   pip install pytorch-lightning==2.6.0 tensorboard==2.20.0 tensorboard-data-server==0.7.2 torchmetrics==1.8.2 lightning-utilities==0.15.2
   ```

4. **Retrieve and Setup PX4-Autopilot (Optional, for online flight)**
   ```bash
   git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
   cd ~/PX4-Autopilot
   bash Tools/setup/ubuntu.sh
   ```

5. **[IMPORTANT] Python Environment Fix for PX4**
   PX4 1.13+ toolchain is incompatible with empy 4.x+. Please rollback to legacy version:
   ```bash
   pip uninstall empy
   pip install empy==3.3.4
   ```
6. Restart the computer after the toolchain is installed, then build PX4 with `make px4_sitl gazebo-classic`.

### Quick Start (Evaluations & Training)

1. **Reproduce Training (Fig 0: Convergence Curves)**
   Run ablation studies to train models and extract convergence data:
   ```bash
   conda activate fcml
   python scripts/offline/run_ablations.py
   python scripts/evaluation/plot_training_curve.py
   ```

2. **Generate T-SNE Evolution (T-SNE Feature Visualization)**
   ```bash
   python scripts/offline/train_offline_tsne.py
   python scripts/evaluation/plot_tsne_astar.py
   ```

3. **Online Flight Evaluation (Fig 1-3: Flight Performance)**
   Requires PX4-Autopilot + Gazebo installed. This runs all 5 controllers × 5 wind conditions = 25 flight tests, then generates comparison figures.
   ```bash
   bash scripts/run_evaluations_mission.sh
   ```

4. **Feature Clustering Analysis (Fig 4-5)**
   ```bash
   python scripts/evaluation/visualize_feature_clusters.py
   ```

5. **Ki Parameter Sweep (Control Parameter Analysis)**
   ```bash
   python scripts/evaluation/simulate_Ki_sweep.py
   ```

### Data Pipeline Configuration

All project paths are managed centrally in `config.py`. By default, paths are relative to the project root. You only need to modify `config.py` if your PX4-Autopilot is installed in a non-default location, or use:
```bash
export PX4_DIR=/path/to/your/PX4-Autopilot
```

**Extracting Data from Raw Flight Logs**
If you have raw PX4 `.ulg` flight logs, convert them to CSV using `pyulog`, then process the CSV files into the training format:
```bash
python scripts/missions/mission_collect.py
```

**Generating DTW Triplets**
After data processing:
```bash
python scripts/alignment/generate_dtw_triplets.py
```

## Directory Structure

```
FCML
├── config.py                   # Centralized path configuration
├── setup.sh                    # One-line quick setup script
├── environment.yml             # Conda environment specification
├── readme.md                   # This document
│
├── checkpoints                 # Trained model weights
│   ├── best_model.pth          # Our method (FCML) best model
│   └── neural_fly_daiml_best.pth  # Baseline Neural-Fly (DAIML) model
│
├── tsne_checkpoints            # T-SNE evolution snapshot models
├── dtw_triplets_data           # DTW-Triplet training dataset
├── processed_data              # Preprocessed flight data (50Hz CSV)
├── raw_logs                    # Raw PX4 flight logs (ULG + CSV)
├── eval_results                # Online evaluation data
├── training_results            # Training curves and intermediate data
├── figures                     # Generated figures (output directory)
│
├── docs                        # Additional documentation
│   └── online_control_architecture.md
│
├── baseline_test               # Native PX4 waypoint mode baselines
│   └── ...
│
└── scripts                     # Core Python and Bash scripts
    ├── alignment               # Data alignment tools (generate_dtw_triplets.py)
    ├── evaluation              # Plotting and analysis scripts
    ├── missions                # Online flight control scripts
    ├── offline                 # Model training scripts
    ├── run_evaluations_mission.sh # Automated flight evaluation
    ├── auto_collect_split.sh   # Automated data collection
    └── set_wind.sh             # Gazebo wind configuration
```

### Naming Conventions

| Code Name      | Description                                |
|----------------|--------------------------------------------|
| `Ours` / `FCML`| Our proposed method (DTW-Triplet alignment)|
| `Neural-Fly`   | Original Neural-Fly with DAIML (baseline)  |
| `Baseline`     | Standard PID controller                    |
| `INDI`         | Incremental Nonlinear Dynamic Inversion    |
| `L1`           | L1 Adaptive Control                        |