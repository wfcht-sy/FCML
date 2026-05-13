#!/usr/bin/env bash
# Stage 1: System packages & Git LFS
set -euo pipefail

log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }

read -p "This will install the missing system dependency: Git LFS. Continue? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    log "Installing git-lfs..."
    sudo apt update
    sudo apt install -y git-lfs
    git lfs install
    log "System setup complete."
else
    log "System setup skipped."
fi
