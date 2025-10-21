import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / 'Hellen_model_RN'

for path in (str(REPO_ROOT), str(MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
