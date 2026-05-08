# Model Training & Evaluation Framework

本仓库包含用于离线模型训练、控制算法在线仿真对比以及实验结果可视化的完整代码环境。核心目标是通过模拟不同风扰动下的飞行任务，验证各控制策略及模型的抗风与跟踪性能。

---

## 目录结构

```text
testmodel/
├── checkpoints/                # 训练好的模型权重与检查点
├── dtw_triplets_data/          # 提取及处理完毕的 DTW-Triplet 训练数据集
├── eval_results/               # 在线飞行性能评估的原始评价数据与统计指标
├── figures/                    # 自动生成的各类可视化输出图表（收敛曲线图、对比图等）
├── processed_data/             # 其他离线训练及评估的预处理数据
├── raw_logs/                   # 每次运行测试的底层日志
├── training_results/           # 离线训练的中间过程数据与输出
├── tsne_checkpoints/           # t-SNE 降维分析用到的模型参数
├── tsne_results/               # t-SNE 可视化的中间数据和结果
├── scripts/                    # 核心脚本目录，按功能模块细分：
│   ├── alignment/              # 数据对齐与预处理工具（如 dtwTriplet 生成）
│   ├── evaluation/             # 用于绘制收敛曲线、对比图与t-SNE聚类的评估生成脚本
│   ├── missions/               # 无人机在线仿真与数据采集飞行任务脚本
│   ├── offline/                # 离线核心算法与模型训练（基于PyTorch Lightning等），包含 models.py
│   └── *.sh                    # 各类打包组合（多模型、多风场）的自动化测试入口脚本与数据流配置
└── readme.md                   # 本文档
```
---

## 主要功能

1. **多范式闭环飞行测试**：包含深度结合 Gazebo 仿真的评估循环（如 `online_mission_compare.py` 等）。支持预置不同风场，对 Baseline、L1、INDI、Neural-Fly 及我们的算法等多路逻辑进行全自动化正面交锋。
2. **核心模型架构及训练体系**：在 `models.py` 中隔离了所有基础网络定义；通过 `train_offline_lightning.py` 等提供全精度、超稳态的多阶段脱机训练。
3. **极简的可视化系统**：包括了 `plot_training_curve.py`、`plot_comparison_mission.py`、`visualize_feature_clusters.py`，以及用于控制器参数扫参仿真的 `simulate_Ki_sweep.py`，负责从绘制收敛图、评估参数扫参影响，到隐藏空间的聚类图等全链路数据分析。
4. **高度模块化封装**：将运行指令包裹在特定的 Shell（如 `run_evaluations_mission.sh`）或 Python 命令（如 `run_ablations.py`）中。每步拆解，运行互不干扰，支持按需测试。

---

## 快速上手

1. 确保安装全部基本依赖（如 PyTorch、pandas、MAVSDK 以及 PX4-SITL/Gazebo 基础组件）。
2. **训练模型与获取收敛比对图（图 0）**：
   利用包装好的对比框架，一次执行将会自我对账多轮训练并画出损失曲线。
   ```bash
   python3 scripts/offline/run_ablations.py
   python3 scripts/evaluation/plot_training_curve.py
   ```
3. **执行在线飞行性能评估（图 1、2、3）**：
   本指令依序遍历各风层与各个对照组模型进行模拟，完成后将自动呼叫制图脚本。
   ```bash
   ./scripts/run_evaluations_mission.sh
   # 可选根据您的需求拆分风况与对比目标
   ```
5. **验证隐空间特征聚类结果（图 4）**：
   在获得了一定的训练与在线数据后，随时生成九宫格状态的 t-SNE 聚类地图。
   ```bash
   python3 scripts/evaluation/visualize_feature_clusters.py
   ```
6. **控制参数鲁棒性扫参分析**：
   运行独立的控制理论推导仿真，分析不同 Ki 参数条件下的追踪 RMSE 表现：
   ```bash
   python3 scripts/evaluation/simulate_Ki_sweep.py
   ```

---

## 开发与扩展

- **配置新模型参数与结构**：修改 `models.py` 补充架构定义，在训练及运行端统一引用。针对特定的探底、退火等行为可定制化修改 `train_offline_lightning.py`。
- **添加复合风况条件**：针对 `run_evaluations_mission.sh` 等调用脚本中的 `test_winds` 数组，增设更多的动态或静态风序列组合。
- **自定义绘图输出**：所有的结果收集及绘图位于对应名称的 `plot_*.py` 中。如补充更多柱状、散点、雷达指标可扩写 `plot_comparison_mission.py` 文件内部逻辑。

---

## 维护者

请保持 README 同步更新。当添加新的模块或重构目录结构时，记得调整本说明文档。不要忘记始终检查端到端飞行数据的完备性。

---
*生成于 2026-04，基于仓库现有代码。*