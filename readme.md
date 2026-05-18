# FCML: Offline Training & Online Adaptive Control Framework

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Platform](https://img.shields.io/badge/Platform-Ubuntu_20.04%20%7C%2022.04-orange)
![PX4](https://img.shields.io/badge/PX4-Autopilot-1.13+-green)

This repository contains the complete codebase for offline model training, online adaptive flight control evaluation, and experiment result visualization. The core objective is to validate tracking and wind-rejection performance of various control strategies (Baseline PID, INDI, L1, Neural-Fly, and our proposed FCML method) under simulated wind disturbances.

### Dependencies and Prerequisites

**System Requirements & Tools:**
- **OS**: Ubuntu 20.04 / 22.04 LTS
- **Python**: Python 3.9
- **Flight Controller Simulation**: PX4-Autopilot (version >= 1.13) and Gazebo 11
- **Critical Tool - Git LFS**: Required for pulling large test logs and datasets. Be sure to install and initialize it before cloning or after cloning via `git lfs pull`.

**NOTE:** The provided installation scripts are intended to be run on *clean* Ubuntu LTS installations to prevent conflicts. 

### Automated Setup Scripts

Instead of a monolithic installation, we provide transparent and separate setup shell scripts where user confirmation (`y/n`) is required before performing any system-level operations.

First, retrieve the repository:
```bash
git clone https://github.com/wfcht-sy/FCML.git
cd FCML
```

**Step 1: Install System Dependencies**
Install standard packages and Git LFS:
```bash
bash setup_system.sh
```

**Step 2: Python / Conda Environment**
Install Miniconda, create the `fcml` environment and pull large LFS data:
```bash
bash setup_env.sh
conda activate fcml
```

**Step 3: Setup PX4 Simulator**
To use simulators for online flight evaluations:
```bash
bash install_px4.sh
```

### Manual Dependency Installation (PyTorch & Additional Tools)

Due to hardware variability, you **must manually install PyTorch** suited to your machine. 

**For CPU-only machines** (e.g., standard servers or basic laptops):
```bash
conda activate fcml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (NVIDIA CUDA) setups**:
Please follow the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to select the exact `cudatoolkit` version matching your hardware. For example:
```bash
conda activate fcml
pip install torch torchvision torchaudio
```

After installing `torch`, you **must** install the logging and training modules:
```bash
pip install pytorch-lightning==2.6.0 tensorboard tensorboardX
```

### Data Extraction from Raw Logs
To reduce repository size, we only keep the compressed `.ulg` raw flight logs. To reproduce plots, you must extract CSV metrics from these logs:
```bash
bash scripts/extract_logs.sh
```

### Quick Start (Evaluations & Training)

1. **Reproduce Training (Fig 0: Convergence Curves)**
   ```bash
   conda activate fcml
   python scripts/offline/run_ablations.py
   python scripts/evaluation/plot_training_curve.py
   ```

2. **Backbone Architecture Ablation (New)**
   Runs a 4-group ablation study to validate the 4-layer network and Triplet Loss design against the Neural-Fly baseline:
   ```bash
   conda activate fcml
   python scripts/offline/run_backbone_ablation.py
   python scripts/evaluation/plot_backbone_ablation.py
   ```

3. **Generate T-SNE Evolution (T-SNE Feature Visualization)**
   ```bash
   conda activate fcml
   python scripts/offline/train_offline_tsne.py
   python scripts/evaluation/plot_tsne_astar.py
   ```

3. **Online Flight Evaluation (Fig 1-3: Flight Performance)**
   Runs missions under various wind speeds:
   ```bash
   conda activate fcml
   bash scripts/run_evaluations_mission.sh
   ```

4. **Feature Clustering Analysis (Fig 4-5)**
   ```bash
   conda activate fcml
   python scripts/evaluation/visualize_feature_clusters.py
   ```

6. **Ki Parameter Sweep (Control Parameter Analysis)**
   Reads simulation metrics to assess optimal integral gain:
   ```bash
   conda activate fcml
   python scripts/evaluation/simulate_Ki_sweep.py
   python scripts/evaluation/plot_Ki_sweep.py
   ```

## Code Refactoring & Improvements
1. **Object-Oriented Controllers**: `online_mission_compare.py` uses a clean `BaseOffboardControl` inheritance model, grouping parameters for Baseline, INDI, L1, and Learning controllers independently.
2. **Component Extraction**: `VirtualWaypointNavigator` and `KinematicSmoother` have been modularized into separate files for better code reuse.
3. **Ablation-Driven Architecture Justification**: Instead of simply unifying the FCML model with Neural-Fly, we prove our 4-layer architectural superiority via the isolated `run_backbone_ablation.py` suite.

## Directory Structure

```
FCML
├── config.py                   # Centralized path configuration
├── setup_system.sh             # Linux system dependencies and Git LFS setup
├── setup_env.sh                # Miniconda environment and dependencies
├── install_px4.sh              # PX4-Autopilot framework installer
├── environment.yml             # Conda environment specification
├── readme.md                   # This document
│
├── evaluate_results            # Simulation evaluation extraction outputs
├── raw_logs                    # Raw ULG flight logs from simulated Gazebo flights
├── scripts                     # Core Python and script execution pipelines
└── ...
```

### Naming Conventions

| Code Name      | Description                                |
|----------------|--------------------------------------------|
| `FCML`         | FCML proposed method (DTW-Triplet alignment)|
| `Neural-Fly`   | Original Neural-Fly with DAIML (baseline)  |
| `Baseline`     | Standard PID controller                    |
| `INDI`         | Incremental Nonlinear Dynamic Inversion    |
