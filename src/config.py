import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RED_WINE_PATH = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_PATH = RAW_DATA_DIR / "winequality-white.csv"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Plot settings
PLOT_SETTINGS = {
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'figsize': (12, 6),
    'fontsize': 10,
    'titlesize': 14,
    'labelsize': 12
}
