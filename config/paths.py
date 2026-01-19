"""
Global path configuration for ENSO_MJO_Tilt project
"""

from pathlib import Path

# ========== ROOT ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ========== DATA ==========
DATA_ROOT = Path(r"E:/Datas")

ERA5_ROOT = DATA_ROOT / "ERA5"
INDEX_ROOT = DATA_ROOT / "ClimateIndex"

# ========== OUTPUT ==========
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIG_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"

# ========== LOG ==========
LOG_ROOT = PROJECT_ROOT / "logs"

# ensure directories exist
for p in [FIG_ROOT, TABLE_ROOT, LOG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)
