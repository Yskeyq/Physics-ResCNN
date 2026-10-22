"""Portable locations; dataset/model files are not covered by the code license."""
import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SOURCE_ROOT.parent
DATA_ROOT = Path(os.environ.get("PHYSICS_RESCNN_DATA", REPOSITORY_ROOT / "data")).expanduser().resolve()
MODELS_ROOT = DATA_ROOT / "models"
RECORDS_ROOT = DATA_ROOT / "records"
OUTPUT_ROOT = Path(os.environ.get("PHYSICS_RESCNN_OUTPUT", REPOSITORY_ROOT / "artifacts")).expanduser().resolve()
