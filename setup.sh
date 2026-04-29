#!/usr/bin/env bash
# =============================================================================
# Project-NFS 一键部署脚本 (Ubuntu 22.04 LTS, x86_64)
#
# 用法：
#   ./setup.sh all                       # 顺序执行 1->5（不含可选的 PX4/Gazebo）
#   ./setup.sh system                    # 阶段 1: 系统基础包
#   ./setup.sh conda                     # 阶段 2: 安装 Miniconda
#   ./setup.sh env                       # 阶段 3: 创建并安装 conda env
#   ./setup.sh link                      # 阶段 4: 兼容绝对路径(/home/zzx/testmodel)
#   ./setup.sh smoke                     # 阶段 5: 离线训练+绘图自检
#   ./setup.sh px4                       # 可选: 安装 PX4-SITL + Gazebo
#   ./setup.sh -h | --help               # 查看帮助
#
# 环境变量(可覆盖默认值)：
#   PROJECT_DIR (默认: $HOME/testmodel)
#   CONDA_HOME  (默认: $HOME/miniconda3)
#   ENV_NAME    (默认: neural-fly      取自 environment.yml 第一行 name:)
#   USE_MAMBA   (默认: 1   设为 0 则使用 conda 解析)
# =============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/testmodel}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-neural-fly}"
USE_MAMBA="${USE_MAMBA:-1}"
LEGACY_LINK="/home/zzx/testmodel"

# ---------- 工具函数 ----------
log()  { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn ]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

activate_conda() {
    # 让脚本内部的 conda activate 生效
    # shellcheck disable=SC1091
    source "$CONDA_HOME/etc/profile.d/conda.sh"
}

# ---------- 阶段实现 ----------
stage_system() {
    log "[1/5] 安装系统级基础包"
    sudo apt update
    sudo apt install -y git curl wget build-essential ca-certificates
}

stage_conda() {
    log "[2/5] 检查 / 安装 Miniconda 至 $CONDA_HOME"
    if [[ -x "$CONDA_HOME/bin/conda" ]]; then
        log "已检测到 conda，跳过安装"
    else
        local installer="/tmp/miniconda.sh"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
        bash "$installer" -b -p "$CONDA_HOME"
        rm -f "$installer"
    fi
    activate_conda
    conda --version
}

stage_env() {
    log "[3/5] 基于 environment.yml 创建 / 更新 conda 环境: $ENV_NAME"
    if [[ ! -f "$PROJECT_DIR/environment.yml" ]]; then
        err "未找到 $PROJECT_DIR/environment.yml — 请先把代码克隆到 $PROJECT_DIR"
        exit 1
    fi
    activate_conda

    local solver="conda"
    if [[ "$USE_MAMBA" == "1" ]]; then
        if ! need_cmd mamba; then
            log "安装 mamba 加速求解 (USE_MAMBA=1)"
            conda install -n base -c conda-forge mamba -y
        fi
        solver="mamba"
    fi

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        log "环境 $ENV_NAME 已存在，执行 update"
        "$solver" env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
    else
        "$solver" env create -f "$PROJECT_DIR/environment.yml"
    fi

    conda activate "$ENV_NAME"
    log "依赖自检"
    python - <<'PY'
import torch, pytorch_lightning, mavsdk, pandas, sklearn
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"lightning={pytorch_lightning.__version__}  mavsdk={mavsdk.__version__}")
PY
}

stage_link() {
    log "[4/5] 兼容脚本里硬编码的 $LEGACY_LINK"
    if [[ "$PROJECT_DIR" == "$LEGACY_LINK" ]]; then
        log "PROJECT_DIR 即为 $LEGACY_LINK，无需软链"
        return 0
    fi
    if [[ -e "$LEGACY_LINK" || -L "$LEGACY_LINK" ]]; then
        warn "$LEGACY_LINK 已存在，跳过软链 (如需重建请手动删除)"
        return 0
    fi
    sudo mkdir -p "$(dirname "$LEGACY_LINK")"
    sudo ln -s "$PROJECT_DIR" "$LEGACY_LINK"
    log "已建立软链 $LEGACY_LINK -> $PROJECT_DIR"
}

stage_smoke() {
    log "[5/5] 离线流水线自检 (训练 + 绘图)"
    activate_conda
    conda activate "$ENV_NAME"
    cd "$PROJECT_DIR"

    log "(a) 训练 Ours 三阶段权重"
    python3 scripts/offline/train_offline_tsne.py

    log "(b) 训练原版 Neural-Fly 基线"
    python3 scripts/offline/train_original_nf_daiml.py

    log "(c) 生成 T-SNE 演化图"
    python3 scripts/evaluation/plot_tsne_astar.py

    log "自检完成，输出位于 $PROJECT_DIR/tsne_results/"
}

stage_px4() {
    log "[opt] 安装 PX4-SITL + Gazebo Classic (可选, 仅在线飞行评估需要)"
    sudo apt install -y gazebo libgazebo-dev
    local px4_dir="$HOME/PX4-Autopilot"
    if [[ ! -d "$px4_dir" ]]; then
        git clone https://github.com/PX4/PX4-Autopilot.git --recursive "$px4_dir"
    fi
    bash "$px4_dir/Tools/setup/ubuntu.sh"
    ( cd "$px4_dir" && make px4_sitl gazebo-classic )
    log "PX4-SITL 编译完成。运行飞行评估前需先在另一终端启动: cd $px4_dir && make px4_sitl gazebo-classic"
}

usage() {
    sed -n '2,20p' "$0"
}

# ---------- 入口 ----------
if [[ $# -eq 0 ]]; then usage; exit 1; fi

case "$1" in
    system) stage_system ;;
    conda)  stage_conda ;;
    env)    stage_env ;;
    link)   stage_link ;;
    smoke)  stage_smoke ;;
    px4)    stage_px4 ;;
    all)
        stage_system
        stage_conda
        stage_env
        stage_link
        stage_smoke
        log "全流程结束。如需在线飞行评估，请额外执行: ./setup.sh px4"
        ;;
    -h|--help|help) usage ;;
    *) err "未知阶段: $1"; usage; exit 1 ;;
esac
