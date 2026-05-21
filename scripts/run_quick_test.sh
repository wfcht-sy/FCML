#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural-fly

WIND="35wind"
python scripts/missions/online_mission_compare.py --controller Baseline --wind $WIND
python scripts/missions/online_mission_compare.py --controller INDI --wind $WIND
python scripts/missions/online_mission_compare.py --controller L1 --wind $WIND
python scripts/missions/online_mission_compare.py --controller Neural-Fly --wind $WIND
python scripts/missions/online_mission_compare.py --controller FCML --wind $WIND

python scripts/evaluation/plot_comparison_mission.py
