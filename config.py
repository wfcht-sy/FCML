import os

# ---------------------------------------------------------------------------
# Project root  (the directory that contains this config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
RAW_LOGS_DIR     = os.path.join(PROJECT_ROOT, "raw_logs")
PROCESSED_DIR    = os.path.join(PROJECT_ROOT, "processed_data")
DTW_DATA_DIR     = os.path.join(PROJECT_ROOT, "dtw_triplets_data")

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
TSNE_CKPT_DIR    = os.path.join(PROJECT_ROOT, "tsne_checkpoints")

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval_results")
TRAINING_DIR     = os.path.join(PROJECT_ROOT, "training_results")
FIGURES_DIR      = os.path.join(PROJECT_ROOT, "figures")
TSNE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "tsne_results")

# ---------------------------------------------------------------------------
# Commonly referenced files
# ---------------------------------------------------------------------------
DTW_CSV          = os.path.join(DTW_DATA_DIR, "dtw_triplet_combined_all.csv")
OURS_MODEL_PATH  = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
NF_MODEL_PATH    = os.path.join(CHECKPOINTS_DIR, "neural_fly_daiml_best.pth")

# ---------------------------------------------------------------------------
# PX4 Autopilot  (override with PX4_DIR environment variable if needed)
# ---------------------------------------------------------------------------
PX4_DIR = os.environ.get(
    "PX4_DIR",
    os.path.join(os.path.expanduser("~"), "PX4-Autopilot")
)
