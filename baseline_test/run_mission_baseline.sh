#!/bin/bash

# ==============================================================================
# PX4 Native Waypoint Mode (Mission Mode) 5 Wind Conditions Baseline Evaluation Script
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WIND_SCRIPT="$PROJECT_ROOT/scripts/set_wind.sh"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 online_mission_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 online_mission_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 online_mission_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 online_mission_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 online_mission_100wind"
)

echo "======================================================"
echo "Starting PX4 Native Waypoint Mode (Mission) Exclusive Evaluation"
echo "======================================================"

for config in "${test_winds[@]}"; do
    read -r wx wy wz gx gy gz wname <<< "$config"
    
    SUCCESS=0
    while [ $SUCCESS -eq 0 ]; do
        echo ">>> [System] Cleaning up environment..."
        pkill -9 px4 2>/dev/null; pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null
        rm -rf ~/.ros/log/* ~/.gazebo/client-* 2>/dev/null
        sleep 2

        echo ">>> [System] Starting PX4 and Gazebo simulation..."
        cd "${PX4_DIR}" || exit
        HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
        for i in {8..1}; do echo -ne "\r Waiting for initialization... $i s "; sleep 1; done
        echo -e "\n"
        cd - > /dev/null

        echo ">>> [Test] Current wind field: ${wname} (Avg wind speed ${wx} m/s)"
        env LD_LIBRARY_PATH="" bash "${WIND_SCRIPT}" $wx $wy $wz $gx $gy $gz
        
        python3 baseline_mission_flight.py --wind "${wname}"
        
        if [ $? -eq 0 ]; then
            echo ">>> Successfully completed current wind condition test."
            SUCCESS=1
            sleep 2
        else
            echo ">>> Detected lag or timeout, retrying this condition..."
        fi
    done
done

echo "All flights completed. Generating exclusive trajectory subplots..."
python3 plot_baseline_only.py