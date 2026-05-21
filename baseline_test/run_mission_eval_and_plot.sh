#!/bin/bash

# ==========================================
# 自动化评估与图表生成工作流
# Automated Evaluation and Plotting Workflow
# ==========================================

CONTROLLERS=("Baseline" "INDI" "L1" "Neural-Fly" "FCML")
WINDS=("nowind" "35wind" "70wind" "100wind")

echo "=========================================="
echo "    启动批量在线仿真任务 (任务数: 20)"
echo "=========================================="

for wind in "${WINDS[@]}"; do
    for ctrl in "${CONTROLLERS[@]}"; do
        echo ">>> 测试执行: 控制器 = ${ctrl}, 风速场景 = ${wind}"
        python scripts/missions/online_mission_compare.py --controller "${ctrl}" --wind "${wind}"
        echo "<<< 测试完成: ${ctrl} @ ${wind}"
        sleep 5  # 增加延迟，确保 MAVSDK 完全断开连接且 PX4 稳定
    done
done

echo ""
echo "=========================================="
echo "    在线飞行测试完毕，开始绘制性能网格图"
echo "=========================================="

python baseline_test/plot_grid.py

echo "可视化绘图已完成。最终图表存放在 figures/ 目录中。"
