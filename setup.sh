#!/usr/bin/env bash
# =============================================================================
# Project-NFS One-Click Deployment Script (Ubuntu 22.04 LTS, x86_64)
#
# Usage:
#   ./setup.sh all                       # Execute sequentially 1->5 (excluding optional PX4/Gazebo)
#   ./setup.sh system                    # Stage 1: System-level basic packages
#   ./setup.sh conda                     # Stage 2: Install Miniconda
#   ./setup.sh env                       # Stage 3: Create and install conda env
#   ./setup.sh link                      # Stage 4: Compatible absolute path (optional legacy symlink)
#   ./setup.sh smoke                     # Stage 5: Offline training + plotting self-check
#   ./setup.sh px4                       # Optional: Install PX4-SITL + Gazebo
#   ./setup.sh -h | --help               # Show help
#
# Environment variables (override defaults):
#   PROJECT_DIR (Default: $HOME/testmodel)
#   CONDA_HOME  (Default: $HOME/miniconda3)
#   ENV_NAME    (Default: neural-fly      extracted from environment.yml name:)
#   USE_MAMBA   (Default: 1   Set to 0 to use conda solver)
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/testmodel}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-neural-fly}"
USE_MAMBA="${USE_MAMBA:-1}"
LEGACY_LINK="${LEGACY_LINK:-}"  # Set via env var if legacy symlink is needed

# ---------- Utility Functions ----------
log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

activate_conda() {
    # Ensure conda activate works inside the script
    # shellcheck disable=SC1091
    source "$CONDA_HOME/etc/profile.d/conda.sh"
}

# ---------- Stage Implementations ----------
stage_system() {
    log "[1/5] Installing system-level basic packages"
    sudo apt update
    sudo apt install -y git curl wget build-essential ca-certificates
}

stage_conda() {
    log "[2/5] Checking / Installing Miniconda to $CONDA_HOME"
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
    log "[3/5] Creating / Updating conda environment based on environment.yml: $ENV_NAME"
    if [[ ! -f "$PROJECT_DIR/environment.yml" ]]; then
        err "Did not find $PROJECT_DIR/environment.yml — Please clone the code to $PROJECT_DIR first"
        exit 1
    fi
    activate_conda

    local solver="conda"
    if [[ "$USE_MAMBA" == "1" ]]; then
        if ! need_cmd mamba; then
            log "Installing mamba to accelerate solving (USE_MAMBA=1)"
            conda install -n base -c conda-forge mamba -y
        fi
        solver="mamba"
    fi

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "Environment $ENV_NAME already exists, running update"
        "$solver" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
    else
        "$solver" env create -f "$PROJECT_DIR/environment.yml"
    fi

    conda activate "$ENV_NAME"
    log "Dependency self-check"
    python - <<'PY'
import torch, pytorch_lightning, mavsdk, pandas, sklearn
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"lightning={pytorch_lightning.__version__}  mavsdk={mavsdk.__version__}")
PY
}

stage_link() {
    log "[4/5] Handling hardcoded paths for backward compatibility (LEGACY_LINK)"
    if [[ -z "$LEGACY_LINK" ]]; then
        log "LEGACY_LINK is empty, skipping."
        return 0
    fi
    if [[ "$PROJECT_DIR" == "$LEGACY_LINK" ]]; then
        log "PROJECT_DIR is already $LEGACY_LINK, no symlink needed"
        return 0
    fi
    if [[ -e "$LEGACY_LINK" || -L "$LEGACY_LINK" ]]; then
        warn "$LEGACY_LINK already exists, skipping symlink (delete manually to recreate)"
        return 0
    fi
    sudo mkdir -p "$(dirname "$LEGACY_LINK")"
    sudo ln -s "$PROJECT_DIR" "$LEGACY_LINK"
    log "Symlink created $LEGACY_LINK -> $PROJECT_DIR"
}

stage_smoke() {
    log "[5/5] Offline pipeline self-check (Training + Plotting)"
    activate_conda
    conda activate "$ENV_NAME"
    cd "$PROJECT_DIR"

    log "(a) Training Ours three-stage weights"
    python3 scripts/offline/train_offline_tsne.py

    log "(b) Training Original Neural-Fly Baseline"
    python3 scripts/offline/train_original_nf_daiml.py

    log "(c) Generating T-SNE evolution plot"
    python3 scripts/evaluation/plot_tsne_astar.py

    log "Self-check completed, output located at $PROJECT_DIR/tsne_results/ or figures/"
}

stage_px4() {
    log "[opt] Installing PX4-SITL + Gazebo Classic (Optional, only needed for online flight evaluations)"
    sudo apt install -y gazebo libgazebo-dev
    local px4_dir="$HOME/PX4-Autopilot"
    if [[ ! -d "$px4_dir" ]]; then
        git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$px4_dir"
    fi
    bash "$px4_dir/Tools/setup/ubuntu.sh"
    ( cd "$px4_dir" && make px4_sitl gazebo-classic )
    log "PX4-SITL compilation completed. Before running flight evaluations, launch it in another terminal: cd $px4_dir && make px4_sitl gazebo-classic"
}

usage() {
    sed -n '2,20p' "$0"
}

# ---------- Entry Point ----------
if [[ $# -eq 0 ]]; then usage; exit 1; fi

case "$1" in
    system) stage_system ;;
    conda)  stage_conda ;;
    env)    stage_env ;;
    link)   stage_link ;;
    smoke)  stage_smoke ;;
    px4)    stage_px4 ;;
    all)
        stage_system
        stage_conda
        stage_env
        stage_link
        stage_smoke
        log "Full process ended. For online flight evaluation, additionally run: ./setup.sh px4"
        ;;
    -h|--help|help) usage ;;
    *) err "Unknown stage: $1"; usage; exit 1 ;;
esac
