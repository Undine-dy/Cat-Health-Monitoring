from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = BACKEND_DIR / "models"
FUSION_MODEL_DIR = MODEL_DIR / "fusion"

PAMAP2_ROOT = DATASET_DIR / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset"
WESAD_ROOT = DATASET_DIR / "WESAD_Kaggle" / "WESAD"
UCI_HAR_ROOT = DATASET_DIR / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset" / "UCIHARDataset"

FUSION_HOST = os.getenv("FUSION_HOST", "127.0.0.1")
FUSION_PORT = int(os.getenv("FUSION_PORT", "8011"))
FUSION_STRESS_DEMO_PORT = int(os.getenv("FUSION_STRESS_DEMO_PORT", "8012"))

LEGACY_HOST = os.getenv("LEGACY_HOST", "0.0.0.0")
LEGACY_PORT = int(os.getenv("LEGACY_PORT", "8000"))
