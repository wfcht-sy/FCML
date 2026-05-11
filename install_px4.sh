#!/bin/bash
# =============================================================================
# PX4-Autopilot One-Click Installation Script
#
# Usage:
#   bash install_px4.sh              # Install with default folder name
#   bash install_px4.sh <folder>     # Install to a custom folder
#
# Tested on Ubuntu 20.04 / 22.04 LTS, x86_64
# =============================================================================
set -euo pipefail

FOLDER_NAME="${1:-PX4-Autopilot}"

SKIP_DOWNLOAD=false

# 1. Check if folder already exists
if [ -d "$FOLDER_NAME" ] && [ "$(ls -A "$FOLDER_NAME" 2>/dev/null)" ]; then
    echo "Directory '$FOLDER_NAME' already exists and is not empty."
    read -p "Do you want to REMOVE it and start fresh? (y/n/q to quit): " choice
    case "$choice" in
        [Yy]* )
            echo "Removing existing directory..."
            rm -rf "$FOLDER_NAME"
            mkdir "$FOLDER_NAME"
            ;;
        [Nn]* )
            echo "Skipping download step, using existing folder."
            SKIP_DOWNLOAD=true
            ;;
        * )
            echo "Exiting script."
            exit 1
            ;;
    esac
else
    mkdir -p "$FOLDER_NAME"
fi

# 2. Download PX4 source code
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Cloning PX4-Autopilot (latest stable)..."
    git clone --recursive https://github.com/PX4/PX4-Autopilot.git "$FOLDER_NAME"
fi

# 3. Install toolchain dependencies
cd "$FOLDER_NAME" || exit
echo "Installing PX4 dependencies..."
bash Tools/setup/ubuntu.sh

# 4. Python compatibility fix
echo "Adjusting empy version for compatibility..."
pip uninstall -y empy 2>/dev/null || true
pip install empy==3.3.4

# 5. Final reminder
echo "----------------------------------------------------------------"
echo "PX4-Autopilot setup complete!"
echo ""
echo "IMPORTANT: Please REBOOT your computer before attempting to build."
echo "This ensures that user group changes and environment variables take effect."
echo ""
echo "After rebooting, verify the installation with:"
echo "  cd $FOLDER_NAME && make px4_sitl gazebo-classic"
echo "----------------------------------------------------------------"
