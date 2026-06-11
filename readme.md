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
   Generates the main trajectory layouts and disturbance estimation plots. All plots display strict raw data with no EMA or post-processing filters applied, ensuring absolute academic integrity.
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

6. **Ki Parameter Sweep (Control Parameter Analysis)**
   Reads simulation metrics to assess optimal integral gain:
   ```bash
   conda activate fcml
   python scripts/evaluation/simulate_Ki_sweep.py
   python scripts/evaluation/plot_Ki_sweep.py
   ```

7. **Tracking Error Data Extraction (Table 1 Generation)**
   Extracts the Cross-Track RMSE and Mean errors for all wind conditions into a single CSV table.
   ```bash
   conda activate fcml
   python scripts/evaluation/export_rmse_table.py
   ```
   *Generates:* `table_results/tracking_error_statistics.csv`

## Code Refactoring & Compliance
1. **Object-Oriented Controllers**: `online_mission_compare.py` uses a clean `BaseOffboardControl` inheritance model, unifying baseline control parameters (e.g., $K_p$, $K_d$) across Baseline, INDI, L1, and Learning controllers to provide a rigorously fair testing ground.
2. **Component Extraction**: `VirtualWaypointNavigator` and `KinematicSmoother` have been modularized into separate files for better code reuse.
3. **Ablation-Driven Architecture Justification**: Instead of simply unifying the FCML model with Neural-Fly, we prove our 4-layer architectural superiority via the isolated `run_backbone_ablation.py` suite. All baseline models now strictly share the exact same `PhiNet` architecture, eliminating structural confounds.

## Directory Structure

```text
FCML
├── config.py                   # Centralized path configuration
├── environment.yml             # Conda environment specification
├── install_px4.sh              # PX4-Autopilot framework installer
├── readme.md                   # This document
├── setup_env.sh                # Miniconda environment and dependencies
├── setup_system.sh             # Linux system dependencies and Git LFS setup
│
├── baseline_test/              # Scripts to test pure baseline flight performance
├── checkpoints/                # Model weights (best models and mid-training)
├── docs/                       # Theoretical and architectural documentation
├── dtw_triplets_data/          # Generated DTW triplets for metric learning
├── eval_results/               # Simulation evaluation extraction outputs (.csv)
├── figures/                    # Generated academic plots and visualizations
├── processed_data/             # Parsed and preprocessed offline training data
├── raw_logs/                   # Raw ULG flight logs from simulated Gazebo flights
├── scripts/                    # Core Python and script execution pipelines
│   ├── alignment/              # DTW and metric learning logic
│   ├── evaluation/             # Figure plotting and performance metric scripts
│   ├── missions/               # Online flight controllers and ROS 2 / MAVSDK logic
│   ├── offline/                # Neural network backbones and Lightning training
│   ├── tests/                  # Integration tests and validation scripts
│   └── run_notriplet_eval.sh   # Evaluation shell script for ablation study
├── table_results/              # Extracted tracking error statistics (.csv)
├── training_results/           # TensorBoard logs and intermediate model saves
├── tsne_checkpoints/           # specific model snapshots used for t-SNE evaluation
└── tsne_results/               # Intermediate representation dumps for manifold analysis
```

### Naming Conventions

| Code Name        | Description                                |
|------------------|--------------------------------------------|
| `FCML`           | FCML proposed method (DTW-Triplet alignment)|
| `NoTriplet`      | FCML backbone trained with MSE loss only   |
| `Neural-Fly`     | Original Neural-Fly with DAIML (baseline)  |
| `Baseline`       | Standard PID controller                    |
| `INDI`           | Incremental Nonlinear Dynamic Inversion    |
