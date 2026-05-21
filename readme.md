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

1. **Backbone Ablation & Training Curves (Loss Only)**
   Evaluates validation MSE convergence across identical backbones under different loss functions.
   ```bash
   conda activate fcml
   python scripts/offline/run_backbone_ablation.py
   python scripts/evaluation/plot_backbone_ablation.py
   ```
   *Generates:* `fig_backbone_ablation.png`

2. **Generate T-SNE Evolution (2x3 Grid Visualization)**
   Visualizes the feature clustering progression (Epoch 0, Mid, Final) comparing Triplet vs non-Triplet.
   ```bash
   conda activate fcml
   python scripts/evaluation/plot_tsne_astar.py
   ```
   *Generates:* `tsne_evolution_final_v2.png`

3. **Online Flight Evaluation (Disturbance Estimation & Trajectories)**
   Generates the main trajectory layouts and disturbance estimation plots (2.0Hz filtered true estimator bounds).
   ```bash
   conda activate fcml
   python scripts/evaluation/plot_comparison_mission.py
   ```
   *Generates:* `fig2_tracking_trajectory_grid.png`, `fig3_disturbance_grid_x/y/z.png`

4. **Online Ablation: Triplet vs MSE (High-density text annotation)**
   Compares the online flight RMSE directly atop the time-series trajectory error curves.
   ```bash
   conda activate fcml
   python scripts/evaluation/compare_online_triplet_vs_mse.py
   ```
   *Generates:* `fig_triplet_vs_mse_online.png`

5. **Ki Parameter Sweep (Control Parameter Analysis)**
   Reads simulation metrics to assess optimal integral gain:
   ```bash
   conda activate fcml
   python scripts/evaluation/simulate_Ki_sweep.py
   python scripts/evaluation/plot_Ki_sweep.py
   ```

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
