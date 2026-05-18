#!/bin/bash
# 一键运行 FCML_NoTriplet 在线飞行评估脚本
# 运行前请确保 PX4 仿真环境 (SITL) 已经在后台运行并处于待机状态

set -e  # 遇到错误即停止

echo "=========================================================="
echo "  开始运行 FCML_NoTriplet (纯 MSE) 在线消融评估"
echo "=========================================================="

# 1. 将离线训练好的无 Triplet 模型复制到 checkpoints 目录
echo "[1/2] 正在复制模型权重..."
cp training_results/backbone_ablation/run_ours_no_triplet/best_model.pth checkpoints/fcml_notriplet.pth
echo "模型已复制为 checkpoints/fcml_notriplet.pth"

# 2. 依次运行五种风况的在线飞行
WINDS=("nowind" "35wind" "70wind" "100wind" "70p20sint")

echo "[2/2] 开始按风况进行在线飞行测试..."
for wind in "${WINDS[@]}"; do
    echo "----------------------------------------------------------"
    echo "  正在测试风况: $wind"
    echo "----------------------------------------------------------"
    python scripts/missions/online_mission_compare.py --controller FCML_NoTriplet --wind "$wind"
    
    # 每次飞行结束后暂停一下，等待仿真器和 MAVSDK 断开连接并重置端口 (UDP 14540)
    echo "  [$wind] 测试完成，等待 10 秒以便仿真器重置和端口释放..."
    sleep 10
done

echo "=========================================================="
echo "  所有 5 种风况的 FCML_NoTriplet 评估均已完成！"
echo "  您现在可以运行画图脚本来查看对比结果："
echo "  python scripts/evaluation/compare_online_triplet_vs_mse.py"
echo "=========================================================="
