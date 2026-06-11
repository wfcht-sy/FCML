#!/bin/bash
# One-click evaluation script for NoTriplet online flight
# Ensure PX4 SITL is running in the background and standing by before running

set -e  # Stop on error

# Ensure execution from project root
cd "$(dirname "$0")/.."
echo "=========================================================="
echo "  Starting NoTriplet (MSE only) online ablation evaluation"
echo "=========================================================="

# 1. Copy the offline trained model without Triplet to checkpoints directory
echo "[1/2] Copying model weights..."
cp training_results/backbone_ablation/run_ours_no_triplet/best_model.pth checkpoints/notriplet.pth
echo "Model copied to checkpoints/notriplet.pth"

# 2. Run online flight for five wind conditions sequentially
WINDS=("nowind" "35wind" "70wind" "100wind" "70p20sint")

echo "[2/2] Starting online flight tests by wind condition..."
for wind in "${WINDS[@]}"; do
    echo "----------------------------------------------------------"
    echo "  Testing wind condition: $wind"
    echo "----------------------------------------------------------"
    python scripts/missions/online_mission_compare.py --controller NoTriplet --wind "$wind"
    
    # Pause after each flight to allow simulator and MAVSDK to disconnect and reset ports (UDP 14540)
    echo "  [$wind] Test completed, waiting 10 seconds for simulator reset and port release..."
    sleep 10
done

echo "=========================================================="
echo "  NoTriplet evaluation for all 5 wind conditions completed!"
echo "  You can now run the plotting script to view the comparison results:"
echo "  python scripts/evaluation/compare_online_triplet_vs_mse.py"
echo "=========================================================="
