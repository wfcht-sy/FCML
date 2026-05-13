#!/usr/bin/env bash
# Stage 2: Conda / Python Environment setup
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-fcml}"

log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

activate_conda() {
    source "$CONDA_HOME/etc/profile.d/conda.sh"
}

read -p "Install/Verify Miniconda and create Conda environment '$ENV_NAME'? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    log "Checking / Installing Miniconda to $CONDA_HOME"
    if [[ -x "$CONDA_HOME/bin/conda" ]]; then
        log "conda detected, skipping installation"
    else
        local installer="/tmp/miniconda.sh"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
        bash "$installer" -b -p "$CONDA_HOME"
        rm -f "$installer"
    fi
    activate_conda

    log "Creating conda environment: $ENV_NAME"
    if [[ ! -f "$PROJECT_DIR/environment.yml" ]]; then
        err "Did not find $PROJECT_DIR/environment.yml"
        exit 1
    fi

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "Environment $ENV_NAME already exists, updating..."
        conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
    else
        conda env create -f "$PROJECT_DIR/environment.yml"
    fi

    log "Python environment prepared. (Please note you need to manually install PyTorch according to your Hardware, see README)."
    
    log "Pulling large dataset files via Git LFS..."
    cd "$PROJECT_DIR"
    git lfs pull
    
    log "Environment setup completed."
else
    log "Environment setup skipped."
fi
