#!/bin/bash

# ==============================================================================
# PX4 原生航点模式 (Mission Mode) 5种风况基线评估脚本
# ==============================================================================

WIND_SCRIPT="/home/zzx/testmodel/Simulated_Data_Autocollect/set_wind.sh"
PX4_DIR="/home/zzx/PX4-Autopilot"

declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 online_mission_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 online_mission_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 online_mission_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 online_mission_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 online_mission_100wind"
)

echo "======================================================"
echo "🚀 开始 PX4 原生航点模式 (Mission) 专属评估"
echo "======================================================"

for config in "${test_winds[@]}"; do
    read -r wx wy wz gx gy gz wname <<< "$config"
    
    SUCCESS=0
    while [ $SUCCESS -eq 0 ]; do
        echo ">>> [系统] 清理环境..."
        pkill -9 px4 2>/dev/null; pkill -9 gzserver 2>/dev/null; pkill -9 gzclient 2>/dev/null
        rm -rf ~/.ros/log/* ~/.gazebo/client-* 2>/dev/null
        sleep 2

        echo ">>> [系统] 启动 PX4 与 Gazebo 仿真..."
        cd "${PX4_DIR}" || exit
        HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
        for i in {8..1}; do echo -ne "\r 等待初始化... $i 秒 "; sleep 1; done
        echo -e "\n"
        cd - > /dev/null

        echo ">>> [测试] 当前风场: ${wname} (平均风速 ${wx} m/s)"
        env LD_LIBRARY_PATH="" bash "${WIND_SCRIPT}" $wx $wy $wz $gx $gy $gz
        
        python3 baseline_mission_flight.py --wind "${wname}"
        
        if [ $? -eq 0 ]; then
            echo ">>> ✅ 当前风况测试成功完成。"
            SUCCESS=1
            sleep 2
        else
            echo ">>> ⚠️ 发现卡顿或超时，正在重试该工况..."
        fi
    done
done

echo "🎉 飞行全部结束，正在生成专属轨迹分图..."
python3 plot_baseline_only.py