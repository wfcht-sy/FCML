#!/bin/bash

cd "$(dirname "$0")/.." || exit


# ==============================================================================
# 虚拟航点模式 (Virtual Mission Mode) 5线控制算法终极评估脚本
# 核心修复: 加入 timeout 进程级熔断与 mavsdk_server 彻底清理，杜绝 Gazebo 僵尸死锁
# ==============================================================================

WIND_SCRIPT="/home/zzx/testmodel/scripts/set_wind.sh"
PX4_DIR="/home/zzx/PX4-Autopilot"

# 5 种风场环境
declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 online_test_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 online_test_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 online_test_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 online_test_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 online_test_100wind"
)

# 5 种对比算法
CONTROLLERS=("Baseline" "INDI" "L1" "Neural-Fly" "Ours")

echo "======================================================"
echo "🚀 开始虚拟航点模式下 5 种算法的巅峰对抗测试"
echo "======================================================"

for ctrl in "${CONTROLLERS[@]}"; do
    for config in "${test_winds[@]}"; do
        read -r wx wy wz gx gy gz wname <<< "$config"
        
        SUCCESS=0
        while [ $SUCCESS -eq 0 ]; do
            echo "------------------------------------------------------"
            echo ">>> [系统] 正在清理历史进程与缓存..."
            pkill -9 px4 2>/dev/null
            pkill -9 gzserver 2>/dev/null
            pkill -9 gzclient 2>/dev/null
            pkill -9 mavsdk_server 2>/dev/null  # [修复]: 杀掉后台残留的 MAVSDK 通讯进程释放 UDP 14540 端口
            pkill -9 -f scripts/missions/online_mission_compare.py 2>/dev/null
            rm -rf ~/.ros/log/* ~/.gazebo/client-* 2>/dev/null
            sleep 3  # [修复]: 多给 OS 1秒钟释放 Socket 资源

            echo ">>> [系统] 正在后台启动 PX4 与 Gazebo 仿真环境..."
            cd "${PX4_DIR}" || exit
            HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
            
            for i in {10..1}; do echo -ne "\r   等待引擎初始化... 剩余 $i 秒   "; sleep 1; done
            echo -e "\n"
            cd - > /dev/null

            echo ">>> [测试] 当前算法: ${ctrl} | 风况: ${wname}"
            
            # [修复]: 使用 timeout 15 包裹风场设置。如果 Gazebo 是僵尸状态导致 gz 卡死，15秒后自动掐断并重试！
            timeout 15 env LD_LIBRARY_PATH="" bash "${WIND_SCRIPT}" $wx $wy $wz $gx $gy $gz
            if [ $? -ne 0 ]; then
                echo ">>> [警告] ⚠️ Gazebo 风场插件无响应或挂起，触发系统熔断，准备重启环境..."
                continue # 直接跳到 while 循环开头，重新清理启动
            fi
            
            # [修复]: 给 Python 进程也加上 120 秒的硬熔断。防止 MAVSDK 内部死锁。
            timeout 120 python3 scripts/missions/online_mission_compare.py --controller "${ctrl}" --wind "${wname}"
            
            if [ $? -eq 0 ]; then
                echo ">>> ✅ 当前工况测试成功完成！"
                SUCCESS=1
                sleep 2
            else
                echo ">>> ⚠️ 发现卡死或超时异常，正在重启该工况..."
            fi
        done
    done
done

echo "🎉 25组飞行全部结束！正在生成论文级对比图表..."
python3 scripts/evaluation/plot_comparison_mission.py