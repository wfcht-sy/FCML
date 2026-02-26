import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 读取日志
csv_path = "checkpoints/training_log.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"错误：找不到文件 {csv_path}")
    print("请确保您是在 'train_offline_with_autocluster.py' 所在的目录下运行此脚本。")
    exit()

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- 画左轴 (Train Task Loss) - 红色实线 ---
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Task Loss (MSE)', color=color)
ax1.plot(df['Epoch'], df['Train_Task'], color=color, linewidth=2, label='Train Task Loss')
ax1.tick_params(axis='y', labelcolor=color)

# 设置 X 轴主刻度间隔为 20
ax1.xaxis.set_major_locator(MultipleLocator(20))

# 打开网格
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# --- 画右轴 (Val Composite Score) - 蓝色实线 ---
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Val Composite Score', color=color)  
# [修改] linestyle='-' 表示实线
ax2.plot(df['Epoch'], df['Val_Composite_Score'], color=color, linewidth=2, linestyle='-', label='Val Score')
ax2.tick_params(axis='y', labelcolor=color)

# 添加标题和布局调整
plt.title("Training Progress Analysis")
fig.tight_layout()

# 保存并显示
plt.savefig("training_curve.png", dpi=300)
print("图表已保存为 training_curve.png")
plt.show()