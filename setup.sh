#!/usr/bin/env bash
# =============================================================================
# Project-NFS (FCML) One-Click Deployment Script
#
# Usage:
#   ./setup.sh all                       # Execute sequentially 1->4 (excluding PX4/Gazebo)
#   ./setup.sh system                    # Stage 1: System packages & Git LFS
#   ./setup.sh conda                     # Stage 2: Install Miniconda
#   ./setup.sh env                       # Stage 3: Create conda env (fcml), install PyTorch & Dataset
#   ./setup.sh smoke                     # Stage 4: Offline training + plotting self-check
#   ./setup.sh px4                       # Optional: Install PX4-SITL + Gazebo
#   ./setup.sh -h | --help               # Show help
#
# Environment variables (override defaults):
#   PROJECT_DIR (Default: auto-detected current dir)
#   CONDA_HOME  (Default: $HOME/miniconda3)
#   ENV_NAME    (Default: fcml)
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-fcml}"

# ---------- Utility Functions ----------
log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

activate_conda() {
    # shellcheck disable=SC1091
    source "$CONDA_HOME/etc/profile.d/conda.sh"
}

# ---------- Stage Implementations ----------
stage_system() {
    log "[1/4] Installing system-level basic packages and Git LFS"
    sudo apt update
    sudo apt install -y git curl wget build-essential ca-certificates git-lfs
    git lfs install
}

stage_conda() {
    log "[2/4] Checking / Installing Miniconda to $CONDA_HOME"
    if [[ -x "$CONDA_HOME/bin/conda" ]]; then
        log "conda detected, skipping installation"
    else
        local installer="/tmp/miniconda.sh"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
        bash "$installer" -b -p "$CONDA_HOME"
        rm -f "$installer"
    fi
    activate_conda
    conda --version
}

stage_env() {
    log "[3/4] Creating conda environment: $ENV_NAME"
    if [[ ! -f "$PROJECT_DIR/environment.yml" ]]; then
        err "Did not find $PROJECT_DIR/environment.yml"
        exit 1
    fi
    activate_conda

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "Environment $ENV_NAME already exists, updating..."
        conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
    else
        conda env create -f "$PROJECT_DIR/environment.yml"
    fi

    conda activate "$ENV_NAME"

    log "Installing PyTorch (CUDA 12.4) and PyTorch Lightning..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install pytorch-lightning==2.6.0 tensorboard tensorboardX

    log "Pulling large dataset files via Git LFS..."
    cd "$PROJECT_DIR"
    git lfs pull

    log "Dependency self-check"
    python - <<'PY'
import torch, pytorch_lightning, pandas, sklearn
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"lightning={pytorch_lightning.__version__}")
PY
}

stage_smoke() {
    log "[4/4] Offline pipeline self-check (Training + Plotting)"
    activate_conda
    conda activate "$ENV_NAME"
    cd "$PROJECT_DIR"

    log "(a) Training Ours three-stage weights"
    python3 scripts/offline/train_offline_tsne.py

    log "(b) Training Original Neural-Fly Baseline"
    python3 scripts/offline/train_original_nf_daiml.py

    log "(c) Generating T-SNE evolution plot"
    python3 scripts/evaluation/plot_tsne_astar.py

    log "Self-check completed, output located at models/results."
}

stage_px4() {
    log "[opt] Installing PX4-SITL + Gazebo Classic (Optional)"
    sudo apt install -y gazebo11 libgazebo11-dev
    local px4_dir="$HOME/PX4-Autopilot"
    if [[ ! -d "$px4_dir" ]]; then
        git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$px4_dir"
    fi
    bash "$px4_dir/Tools/setup/ubuntu.sh"
    ( cd "$px4_dir" && make px4_sitl gazebo-classic )
    conda activate "$ENV_NAME"
    pip uninstall -y empy
    pip install empy==3.3.4
    log "PX4-SITL compilation completed. Launch it via: cd $px4_dir && make px4_sitl gazebo-classic"
}

usage() {
    sed -n '2,14p' "$0"
}

# ---------- Entry Point ----------
if [[ $# -eq 0 ]]; then usage; exit 1; fi

case "$1" in
    system) stage_system ;;
    conda)  stage_conda ;;
    env)    stage_env ;;
    smoke)  stage_smoke ;;
    px4)    stage_px4 ;;
    all)
        stage_system
        stage_conda
        stage_env
        stage_smoke
        log "Full deployment ended. For online flight evaluation, run: ./setup.sh px4"
        ;;
    -h|--help|help) usage ;;
    *) err "Unknown stage: $1"; usage; exit 1 ;;
esac
