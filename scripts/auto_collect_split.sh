#!/bin/bash

# ================= 配置区域 =================
# 训练集: 随机轨迹 + 0-50档位静态风
# 格式: "MX MY MZ GX GY GZ 文件夹名"
declare -a train_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 train_nowind"
    "1.3 0.0 0.0 0.0 0.0 0.0 train_10wind"
    "2.5 0.0 0.0 0.0 0.0 0.0 train_20wind"
    "3.7 0.0 0.0 0.0 0.0 0.0 train_30wind"
    "4.9 0.0 0.0 0.0 0.0 0.0 train_40wind"
    "6.1 0.0 0.0 0.0 0.0 0.0 train_50wind"
)
TRAIN_SEED=12345

# 测试集: 8字轨迹 + 内插/外推/动态风
# 70p20sint: 基础风 8.5, 阵风振幅 2.4
declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 test_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 test_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 test_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 test_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 test_100wind"
)

# [路径配置]
PX4_DIR=~/PX4-Autopilot
BASE_LOG_ROOT="$PX4_DIR/build/px4_sitl_default/rootfs/log"

# 检查依赖
if ! command -v ulog2csv &> /dev/null; then
    echo "错误: 未找到 ulog2csv 命令。"
    echo "请运行: pip install pyulog"
    exit 1
fi

if [ ! -d "$BASE_LOG_ROOT" ]; then
    echo "警告: 找不到日志根目录 $BASE_LOG_ROOT"
    echo "如果是第一次运行，该目录可能尚未创建。"
fi

# ================= 工具函数 =================

# [新增] 写入高频日志话题配置，与 logged_topics.cpp 修改保持一致
write_logger_topics() {
    local log_dir="$BASE_LOG_ROOT"
    mkdir -p "$log_dir"

    cat > "$log_dir/logger_topics.txt" << 'EOF'
# PX4 Logger Topics Override
# 与 logged_topics.cpp 中的修改保持一致
# interval_ms=0 表示全速记录

# ---- 位置 (~50Hz 全速) ----
vehicle_local_position 0
vehicle_local_position_groundtruth 0

# ---- 姿态 (~250Hz 全速) ----
vehicle_attitude 0
vehicle_attitude_groundtruth 0

# ---- 电机输出 (~400Hz 全速) ----
actuator_outputs 0
actuator_motors 0

# ---- 陀螺仪 (~1000Hz 全速) ----
sensor_gyro 0

# ---- 保留低频状态话题，用于基本状态追溯 ----
vehicle_status 1000
vehicle_land_detected 1000
vehicle_control_mode 1000
EOF
    echo "  [配置] logger_topics.txt 已写入: $log_dir/logger_topics.txt"
}

# [新增] 验证 CSV 文件的实际记录频率
verify_csv_frequency() {
    local target_dir=$1
    echo "  [验证] 检查 CSV 实际记录频率..."
    python3 - "$target_dir" << 'PYEOF'
import sys, os, glob
import numpy as np

target_dir = sys.argv[1]
topics = {
    "vehicle_local_position": "~50Hz",
    "vehicle_attitude":       "~250Hz",
    "actuator_outputs":       "~400Hz",
    "sensor_gyro":            "~1000Hz",
}

all_ok = True
for topic, expected in topics.items():
    # CSV 文件名形如 xxx_vehicle_local_position_0.csv
    pattern = os.path.join(target_dir, f"*{topic}*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"  [警告] 未找到 {topic} 的 CSV 文件")
        all_ok = False
        continue

    csv_file = files[0]
    try:
        import pandas as pd
        df = pd.read_csv(csv_file, usecols=['timestamp'], nrows=500)
        if len(df) < 2:
            print(f"  [警告] {topic}: 数据行数不足")
            continue
        ts = df['timestamp'].values / 1e6  # us → s
        dt = np.diff(ts)
        avg_hz = 1.0 / dt.mean()
        max_hz = 1.0 / dt.min()
        print(f"  [OK] {topic}: 均值={avg_hz:.1f}Hz, 峰值={max_hz:.1f}Hz (预期{expected})")
    except Exception as e:
        print(f"  [错误] {topic} 验证失败: {e}")
        all_ok = False

if all_ok:
    print("  [验证通过] 所有话题频率正常")
else:
    print("  [验证失败] 部分话题频率异常，请检查 logger_topics.txt 是否正确加载")
PYEOF
}

wait_for_gazebo() {
    local timeout=60
    local count=0
    local target_topic="mean_wind_cmd"

    echo -n "  [等待] 正在检测风场插件..."

    while [ $count -lt $timeout ]; do
        if gz topic -l 2>/dev/null | grep -q "$target_topic"; then
            echo " [成功] 插件话题已就绪！"
            return 0
        fi
        sleep 2
        count=$((count+2))
    done

    echo ""
    echo "  [错误] 等待超时！请检查 .world 文件是否正确加载了插件。"
    return 1
}

run_collection() {
    local type=$1; local winds_arr=$2; local script=$3; local extra_arg=$4
    local -n winds=$winds_arr

    for scenario in "${winds[@]}"; do
        set -- $scenario
        WIND_ARGS=("$1" "$2" "$3" "$4" "$5" "$6")
        DIR_NAME="$7"

        echo "------------------------------------------------"
        echo ">>> 采集 [$type]: $DIR_NAME"

        # 1. 环境清理
        pkill -x px4; pkill -x gzserver; pkill -x gzclient; pkill -f mavsdk
        killall -9 px4 gzserver gzclient 2>/dev/null
        sleep 2

        if [ -d "$BASE_LOG_ROOT" ]; then
            rm -rf "$BASE_LOG_ROOT"/*
        fi

        # [新增] 清理完成后、启动前写入话题频率配置
        write_logger_topics

        # 2. 启动仿真
        cd $PX4_DIR
        HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
        PX4_PID=$!

        echo "  [系统] 仿真启动中 (等待 10s)..."
        sleep 10

        cd - > /dev/null
        if ! wait_for_gazebo; then
            echo "  [跳过] 插件未加载，重试下一轮..."
            kill -9 $PX4_PID 2>/dev/null
            killall -9 px4 gzserver gzclient 2>/dev/null
            continue
        fi

        sleep 5

        # 3. 运行飞行脚本
        if [ "$type" == "train" ]; then
            python3 $script --wind "${WIND_ARGS[@]}" $extra_arg
        else
            python3 $script "${WIND_ARGS[@]}"
        fi

        # 4. 抓取并重命名日志
        echo "  [日志] 正在归档..."

        LATEST_DATE_DIR=$(ls -td "$BASE_LOG_ROOT"/*/ 2>/dev/null | head -n 1)

        if [ -z "$LATEST_DATE_DIR" ]; then
            echo "  [错误] 未找到日期日志文件夹！"
        else
            LATEST_ULG=$(ls -t "$LATEST_DATE_DIR"*.ulg 2>/dev/null | head -n 1)

            if [ -z "$LATEST_ULG" ]; then
                echo "  [错误] 文件夹为空，未生成 ULG 文件！"
            else
                TARGET_DIR="/home/zzx/testmodel/raw_logs/$DIR_NAME"
                rm -rf "$TARGET_DIR"
                mkdir -p $TARGET_DIR

                TARGET_FILE="$TARGET_DIR/${DIR_NAME}.ulg"

                cp "$LATEST_ULG" "$TARGET_FILE"
                echo "  [归档] 日志已保存为: $TARGET_FILE"

                # [修改] 只提取脚本实际需要的话题
                echo "  [转换] 正在转换为 CSV..."
                ulog2csv "$TARGET_FILE" -o "$TARGET_DIR" > /dev/null 2>&1

                echo "  [完成] CSV 已生成至: $TARGET_DIR"

                # [新增] 验证第一个工况的频率，后续工况跳过以节省时间
                if [ "$DIR_NAME" == "train_nowind" ] || [ "$DIR_NAME" == "test_nowind" ]; then
                    verify_csv_frequency "$TARGET_DIR"
                fi
            fi
        fi

        # 结束本轮 (暴力清理进程树)
        echo "  [系统] 正在关闭并彻底清理仿真进程..."
        kill -9 $PX4_PID 2>/dev/null
        killall -9 px4 gzserver gzclient 2>/dev/null
        sleep 5
    done
}

# ================= 主流程 =================

echo "=== Phase 1: Training Set (Random Spline, 150s) ==="
run_collection "train" train_winds "mission_fly_random.py" "--seed $TRAIN_SEED"

echo "=== Phase 2: Test Set (Figure-8, 90s) ==="
run_collection "test" test_winds "mission_fly_single.py" ""

echo "=== 所有采集任务结束 ==="
echo "数据已按顺序保存至 /home/zzx/testmodel/raw_logs/ 文件夹下"
