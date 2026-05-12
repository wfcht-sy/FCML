#!/bin/bash

# 1. Ask for project folder name
read -p "Enter project folder name [Default: WARA-PX4]: " FOLDER_NAME
FOLDER_NAME=${FOLDER_NAME:-WARA-PX4}

SKIP_DOWNLOAD=false

# 2. Check if folder exists and is not empty
if [ -d "$FOLDER_NAME" ] && [ "$(ls -A "$FOLDER_NAME")" ]; then
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

# 3. Download PX4 source code
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "Cloning PX4-Autopilot v1.13.3..."
    git clone --depth 1 --branch v1.13.3 --recursive https://github.com/PX4/PX4-Autopilot.git "$FOLDER_NAME"
fi

# 4. Apply the patch
cd "$FOLDER_NAME" || exit
if [ -f "../WARA-PX4_v1.13.3.patch" ]; then
    echo "Applying WARA-PX4_v1.13.3.patch..."
    git apply -C1 --ignore-whitespace < "../WARA-PX4_v1.13.3.patch"
else
    echo "Warning: WARA-PX4_v1.13.3.patch not found in the parent directory. Skipping patch."
fi

# 5. Execute ubuntu.sh to install dependencies
echo "Installing PX4 dependencies..."
bash Tools/setup/ubuntu.sh

# 6. Fix empy version
# PX4 v1.13.3 relies on older attributes of the 'empy' library.
# Newer versions of empy (4.x+) removed these legacy features, causing build failures.
# We must roll back to 3.3.4 for compatibility with the v1.13.3 build system.
echo "Adjusting empy version for compatibility..."
pip uninstall -y empy
pip install empy==3.3.4

# 7. Final reminder
echo "----------------------------------------------------------------"
echo "Setup complete!"
echo "IMPORTANT: Please REBOOT your computer before attempting to compile."
echo "This ensures that user group changes (like 'dialout') and environment variables take effect."
echo ""
echo "After rebooting, you can perform a build such as:"
echo "  make px4_sitl_wavelet jmavsim (for simulation)"
echo "  make px4_fmu-v5_wavelet       (for hardware targets)"
echo "----------------------------------------------------------------"
