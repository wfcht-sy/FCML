#!/bin/bash

cd "$(dirname "$0")/.." || exit

# ==============================================================================
# Virtual Mission Mode: 5-Controller Comparative Evaluation Script
# Key fix: added timeout-based process fusing and mavsdk_server cleanup
#          to prevent Gazebo zombie deadlocks
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WIND_SCRIPT="$PROJECT_ROOT/scripts/set_wind.sh"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

# 5 wind conditions
declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 online_test_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 online_test_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 online_test_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 online_test_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 online_test_100wind"
)

# 5 controllers under comparison
CONTROLLERS=("Baseline" "INDI" "L1" "Neural-Fly" "FCML")

# ==============================================================================
# Per-Controller Re-run Configuration
# Set to 'true' to force re-simulation even if data exists.
# Set to 'false' to fast-skip if data already exists.
# ==============================================================================
declare -A RERUN_CONFIG
RERUN_CONFIG=(
    ["Baseline"]=false
    ["INDI"]=false
    ["L1"]=false
    ["Neural-Fly"]=true
    ["FCML"]=false
)

# Optional global override via command line (e.g., bash run_evaluations_mission.sh true)
if [[ "$1" == "true" ]] || [[ "$1" == "--force-rerun" ]]; then
    echo ">>> [Info] Global force re-run override is enabled. All scenarios will be re-simulated."
    for ctrl in "${!RERUN_CONFIG[@]}"; do
        RERUN_CONFIG[$ctrl]=true
    done
fi

echo "======================================================"
echo "Starting Virtual Mission evaluation for 5 controllers"
echo "======================================================"

for ctrl in "${CONTROLLERS[@]}"; do
    for config in "${test_winds[@]}"; do
        read -r wx wy wz gx gy gz wname <<< "$config"
        
        CSV_FILE="${PROJECT_ROOT}/eval_results/eval_data_VirtualMission_${ctrl}_${wname}.csv"
        
        # Check per-controller re-run flag
        if [ "${RERUN_CONFIG[$ctrl]}" = false ] && [ -f "$CSV_FILE" ]; then
            echo ">>> [Skip] Controller: ${ctrl} | Wind: ${wname} (Data already exists)"
            continue
        fi
        
        SUCCESS=0
        while [ $SUCCESS -eq 0 ]; do
            echo "------------------------------------------------------"
            echo ">>> [System] Cleaning up stale processes and caches..."
            pkill -9 px4 2>/dev/null
            pkill -9 gzserver 2>/dev/null
            pkill -9 gzclient 2>/dev/null
            pkill -9 mavsdk_server 2>/dev/null  # Kill lingering MAVSDK to free UDP 14540
            pkill -9 -f scripts/missions/online_mission_compare.py 2>/dev/null
            rm -rf ~/.ros/log/* ~/.gazebo/client-* 2>/dev/null
            sleep 3  # Allow OS to release socket resources

            echo ">>> [System] Launching PX4 + Gazebo simulation..."
            cd "${PX4_DIR}" || exit
            HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
            
            for i in {10..1}; do echo -ne "\r   Waiting for engine init... $i s remaining   "; sleep 1; done
            echo -e "\n"
            cd - > /dev/null

            echo ">>> [Test] Controller: ${ctrl} | Wind: ${wname}"
            
            # Use timeout to guard against Gazebo zombie hanging the gz command
            timeout 15 env LD_LIBRARY_PATH="" bash "${WIND_SCRIPT}" $wx $wy $wz $gx $gy $gz
            if [ $? -ne 0 ]; then
                echo ">>> [WARN] Gazebo wind plugin unresponsive, triggering restart..."
                continue
            fi
            
            # Hard timeout for the Python process to prevent MAVSDK deadlocks
            timeout 120 python3 scripts/missions/online_mission_compare.py --controller "${ctrl}" --wind "${wname}" --force_rerun
            
            if [ $? -eq 0 ]; then
                echo ">>> [OK] Test case completed successfully!"
                SUCCESS=1
                sleep 2
            else
                echo ">>> [WARN] Deadlock or timeout detected, restarting this test case..."
            fi
        done
    done
done

echo "All 25 flight tests completed. Generating comparison figures..."
python3 scripts/evaluation/plot_comparison_mission.py