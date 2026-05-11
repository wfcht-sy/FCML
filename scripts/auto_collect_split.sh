#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ================= Configuration =================
# Training set: random spline trajectories + 0-50 level static winds
# Format: "MX MY MZ GX GY GZ folder_name"
declare -a train_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 train_nowind"
    "1.3 0.0 0.0 0.0 0.0 0.0 train_10wind"
    "2.5 0.0 0.0 0.0 0.0 0.0 train_20wind"
    "3.7 0.0 0.0 0.0 0.0 0.0 train_30wind"
    "4.9 0.0 0.0 0.0 0.0 0.0 train_40wind"
    "6.1 0.0 0.0 0.0 0.0 0.0 train_50wind"
)
TRAIN_SEED=12345

# Test set: figure-8 trajectories + interpolated/extrapolated/dynamic winds
# 70p20sint: base wind 8.5 m/s, gust amplitude 2.4 m/s
declare -a test_winds=(
    "0.0 0.0 0.0 0.0 0.0 0.0 test_nowind"
    "4.2 0.0 0.0 0.0 0.0 0.0 test_35wind"
    "8.5 0.0 0.0 0.0 0.0 0.0 test_70wind"
    "8.5 0.0 0.0 2.4 0.0 0.0 test_70p20sint"
    "12.1 0.0 0.0 0.0 0.0 0.0 test_100wind"
)

# [Path Configuration]
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BASE_LOG_ROOT="$PX4_DIR/build/px4_sitl_default/rootfs/log"

# Check dependencies
if ! command -v ulog2csv &> /dev/null; then
    echo "Error: ulog2csv command not found."
    echo "Please run: pip install pyulog"
    exit 1
fi

if [ ! -d "$BASE_LOG_ROOT" ]; then
    echo "Warning: log root directory not found: $BASE_LOG_ROOT"
    echo "This directory may not exist until the first simulation run."
fi

# ================= Utility Functions =================

# Write high-frequency logger topic configuration (aligned with logged_topics.cpp)
write_logger_topics() {
    local log_dir="$BASE_LOG_ROOT"
    mkdir -p "$log_dir"

    cat > "$log_dir/logger_topics.txt" << 'EOF'
# PX4 Logger Topics Override
# Aligned with logged_topics.cpp modifications
# interval_ms=0 means full-rate recording

# ---- Position (~50Hz full-rate) ----
vehicle_local_position 0
vehicle_local_position_groundtruth 0

# ---- Attitude (~250Hz full-rate) ----
vehicle_attitude 0
vehicle_attitude_groundtruth 0

# ---- Actuator outputs (~400Hz full-rate) ----
actuator_outputs 0
actuator_motors 0

# ---- Gyroscope (~1000Hz full-rate) ----
sensor_gyro 0

# ---- Low-frequency status topics for basic state tracing ----
vehicle_status 1000
vehicle_land_detected 1000
vehicle_control_mode 1000
EOF
    echo "  [Config] logger_topics.txt written to: $log_dir/logger_topics.txt"
}

# Verify actual recording frequency of CSV files
verify_csv_frequency() {
    local target_dir=$1
    echo "  [Verify] Checking actual CSV recording frequency..."
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
    pattern = os.path.join(target_dir, f"*{topic}*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"  [WARN] CSV file not found for {topic}")
        all_ok = False
        continue

    csv_file = files[0]
    try:
        import pandas as pd
        df = pd.read_csv(csv_file, usecols=['timestamp'], nrows=500)
        if len(df) < 2:
            print(f"  [WARN] {topic}: insufficient data rows")
            continue
        ts = df['timestamp'].values / 1e6  # us -> s
        dt = np.diff(ts)
        avg_hz = 1.0 / dt.mean()
        max_hz = 1.0 / dt.min()
        print(f"  [OK] {topic}: avg={avg_hz:.1f}Hz, peak={max_hz:.1f}Hz (expected {expected})")
    except Exception as e:
        print(f"  [ERROR] {topic} verification failed: {e}")
        all_ok = False

if all_ok:
    print("  [PASS] All topic frequencies are normal")
else:
    print("  [FAIL] Some topic frequencies are abnormal, check logger_topics.txt")
PYEOF
}

wait_for_gazebo() {
    local timeout=60
    local count=0
    local target_topic="mean_wind_cmd"

    echo -n "  [Wait] Detecting wind plugin..."

    while [ $count -lt $timeout ]; do
        if gz topic -l 2>/dev/null | grep -q "$target_topic"; then
            echo " [OK] Plugin topic is ready!"
            return 0
        fi
        sleep 2
        count=$((count+2))
    done

    echo ""
    echo "  [ERROR] Timed out! Check if the .world file loaded the wind plugin correctly."
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
        echo ">>> Collecting [$type]: $DIR_NAME"

        # 1. Environment cleanup
        pkill -x px4; pkill -x gzserver; pkill -x gzclient; pkill -f mavsdk
        killall -9 px4 gzserver gzclient 2>/dev/null
        sleep 2

        if [ -d "$BASE_LOG_ROOT" ]; then
            rm -rf "$BASE_LOG_ROOT"/*
        fi

        # Write logger topic frequency config before launching
        write_logger_topics

        # 2. Launch simulation
        cd $PX4_DIR
        HEADLESS=1 make px4_sitl_default gazebo_iris__windy > /dev/null 2>&1 &
        PX4_PID=$!

        echo "  [System] Simulation starting (waiting 10s)..."
        sleep 10

        cd - > /dev/null
        if ! wait_for_gazebo; then
            echo "  [Skip] Plugin not loaded, retrying next round..."
            kill -9 $PX4_PID 2>/dev/null
            killall -9 px4 gzserver gzclient 2>/dev/null
            continue
        fi

        sleep 5

        # 3. Run flight script
        if [ "$type" == "train" ]; then
            python3 $script --wind "${WIND_ARGS[@]}" $extra_arg
        else
            python3 $script "${WIND_ARGS[@]}"
        fi

        # 4. Archive and rename logs
        echo "  [Log] Archiving..."

        LATEST_DATE_DIR=$(ls -td "$BASE_LOG_ROOT"/*/ 2>/dev/null | head -n 1)

        if [ -z "$LATEST_DATE_DIR" ]; then
            echo "  [ERROR] No date log folder found!"
        else
            LATEST_ULG=$(ls -t "$LATEST_DATE_DIR"*.ulg 2>/dev/null | head -n 1)

            if [ -z "$LATEST_ULG" ]; then
                echo "  [ERROR] Folder is empty, no ULG file generated!"
            else
                TARGET_DIR="$PROJECT_ROOT/raw_logs/$DIR_NAME"
                rm -rf "$TARGET_DIR"
                mkdir -p $TARGET_DIR

                TARGET_FILE="$TARGET_DIR/${DIR_NAME}.ulg"

                cp "$LATEST_ULG" "$TARGET_FILE"
                echo "  [Archive] Log saved as: $TARGET_FILE"

                # Extract only the topics needed by the processing scripts
                echo "  [Convert] Converting to CSV..."
                ulog2csv "$TARGET_FILE" -o "$TARGET_DIR" > /dev/null 2>&1

                echo "  [Done] CSV files generated in: $TARGET_DIR"

                # Verify frequency for the first scenario only (to save time)
                if [ "$DIR_NAME" == "train_nowind" ] || [ "$DIR_NAME" == "test_nowind" ]; then
                    verify_csv_frequency "$TARGET_DIR"
                fi
            fi
        fi

        # End of round: force-kill all simulation processes
        echo "  [System] Shutting down and cleaning simulation processes..."
        kill -9 $PX4_PID 2>/dev/null
        killall -9 px4 gzserver gzclient 2>/dev/null
        sleep 5
    done
}

# ================= Main Pipeline =================

echo "=== Phase 1: Training Set (Random Spline, 150s) ==="
run_collection "train" train_winds "mission_fly_random.py" "--seed $TRAIN_SEED"

echo "=== Phase 2: Test Set (Figure-8, 90s) ==="
run_collection "test" test_winds "mission_fly_single.py" ""

echo "=== All collection tasks completed ==="
echo "Data saved to $PROJECT_ROOT/raw_logs/"
