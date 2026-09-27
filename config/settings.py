
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "bert_imdb_80_20_split_final"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(LOCAL_MODEL_PATH)
)