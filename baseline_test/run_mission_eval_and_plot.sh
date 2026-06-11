#!/bin/bash

# ==========================================
# Automated Evaluation and Plotting Workflow
# ==========================================

CONTROLLERS=("Baseline" "INDI" "L1" "Neural-Fly" "FCML")
WINDS=("nowind" "35wind" "70wind" "100wind")

echo "=========================================="
echo "    Starting Batch Online Simulation (Tasks: 20)"
echo "=========================================="

for wind in "${WINDS[@]}"; do
    for ctrl in "${CONTROLLERS[@]}"; do
        echo ">>> Test execution: Controller = ${ctrl}, Wind condition = ${wind}"
        python scripts/missions/online_mission_compare.py --controller "${ctrl}" --wind "${wind}" --force_rerun
        echo "<<< Test completed: ${ctrl} @ ${wind}"
        sleep 5  # Add delay to ensure MAVSDK fully disconnects and PX4 stabilizes
    done
done

echo ""
echo "=========================================="
echo "    Online flight tests completed, starting plotting"
echo "=========================================="

python scripts/evaluation/plot_comparison_mission.py

echo "Visualizations completed. Final plots are saved in the figures/ directory."
