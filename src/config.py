"""Central configuration: paths, random seed and the chronological split."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"

# Chronological split (inclusive time-step ranges). Elliptic has 49 steps.
TRAIN_STEPS = (1, 27)
VAL_STEPS = (28, 34)
TEST_STEPS = (35, 49)

# Label encoding used in the master table.
ILLICIT, LICIT, UNKNOWN = 1, 0, -1

N_LOCAL, N_AGG = 93, 72
LOCAL_COLS = [f"local_{i}" for i in range(1, N_LOCAL + 1)]
AGG_COLS = [f"agg_{i}" for i in range(1, N_AGG + 1)]
# 165 anonymised features (the time step itself is NOT used as a model input)
RAW_FEATURE_COLS = LOCAL_COLS + AGG_COLS

# Graph/time features added by feature_engineering.py
ENG_COLS_BASE = ["in_degree", "out_degree", "degree", "step_tx_count"]

for _d in (PROC_DIR, MODELS_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
