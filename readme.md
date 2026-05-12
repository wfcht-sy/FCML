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

**IMPORTANT NOTE**
1. We recommend to **create a virtual environment** before proceeding the installation.
2. The installation scripts are intended to be run on *clean* Ubuntu LTS installations.

### One-line Quick Setup

We provide a bash script to setup the project dependencies, conda environment (`fcml`), and PX4-Autopilot toolchains.

First, retrieve the repository and pull all files:
```bash
git clone https://github.com/wfcht-sy/FCML.git
cd FCML
```

Then, run the one-line setup script:
```bash
bash setup.sh all
```

*Note: The `all` command covers system packages, Conda installation, FCML python environment + PyTorch setup, Git LFS data pulling, and an offline pipeline self-check (`smoke`).*

If you plan to run online flight evaluations in Gazebo, also run:
```bash
bash setup.sh px4
```
Reboot the computer to complete the setup.
After that, read the **Quick Start** section to verify the installation.

### Manual Setup Guide
If the auto installation was interrupted, or a manual configuration is preferred, read the following manual setup guide:

1. **Install Git LFS and fetch datasets**
   ```bash
   sudo apt update
   sudo apt install -y git-lfs
   git lfs install
   git lfs pull
   ```

2. **Create the Conda Environment**
   Install Miniconda if you don't have it, then:
   ```bash
   conda env create -f environment.yml
   conda activate fcml
   ```

3. **Install PyTorch & Lightning**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install pytorch-lightning==2.6.0 tensorboard tensorboardX
   ```

4. **Retrieve and Setup PX4-Autopilot (Optional, for online flight)**
   ```bash
   sudo apt install -y gazebo11 libgazebo11-dev
   git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
   cd ~/PX4-Autopilot
   bash Tools/setup/ubuntu.sh
   ```

5. **[IMPORTANT] Python Environment Fix for PX4**
   PX4 1.13+ toolchain is incompatible with empy 4.x+. Please rollback to legacy version:
   ```bash
   conda activate fcml
   pip uninstall -y empy
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
   conda activate fcml
   python scripts/offline/train_offline_tsne.py
   python scripts/evaluation/plot_tsne_astar.py
   ```

3. **Online Flight Evaluation (Fig 1-3: Flight Performance)**
   Requires PX4-Autopilot + Gazebo installed. This runs all 5 controllers × 5 wind conditions = 25 flight tests, then generates comparison figures.
   ```bash
   conda activate fcml
   bash scripts/run_evaluations_mission.sh
   ```

4. **Feature Clustering Analysis (Fig 4-5)**
   ```bash
   conda activate fcml
   python scripts/evaluation/visualize_feature_clusters.py
   ```

5. **Ki Parameter Sweep (Control Parameter Analysis)**
   ```bash
   conda activate fcml
   python scripts/evaluation/simulate_Ki_sweep.py
   ```

### Data Pipeline Configuration

All project paths are managed centrally in `config.py`. By default, paths are relative to the project root. You only need to modify `config.py` if your PX4-Autopilot is installed in a non-default location, or use:
```bash
export PX4_DIR=/path/to/your/PX4-Autopilot
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
├── dtw_triplets_data           # DTW-Triplet training dataset
├── scripts                     # Core Python and Bash scripts
└── ...
```

### Naming Conventions

| Code Name      | Description                                |
|----------------|--------------------------------------------|
| `Ours` / `FCML`| Our proposed method (DTW-Triplet alignment)|
| `Neural-Fly`   | Original Neural-Fly with DAIML (baseline)  |
| `Baseline`     | Standard PID controller                    |
| `INDI`         | Incremental Nonlinear Dynamic Inversion    |
