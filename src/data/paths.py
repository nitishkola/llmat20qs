"""Canonical data locations used by experiment runners."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = RAW_DATA_DIR / "results"
TARGET_MANIFEST = PROCESSED_DATA_DIR / "target_pool_v1.json"
