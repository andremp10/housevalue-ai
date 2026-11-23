"""Configuração central do pipeline de ML do HouseValue AI.

Todos os caminhos são relativos à raiz do projeto para que os scripts
funcionem independentemente do diretório de execução.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Artefatos principais
DATA_PROFILE_PATH = ARTIFACTS_DIR / "data_profile.json"
DATA_DICTIONARY_PATH = ARTIFACTS_DIR / "data_dictionary.json"
SOURCE_MANIFEST_PATH = DATA_DIR / "source_manifest.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
SPLIT_PATH = METRICS_DIR / "split.json"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.json"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.json"
HOLDOUT_REPORT_PATH = METRICS_DIR / "holdout_evaluation.json"
APP_MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"
FULL_MODEL_PATH = MODELS_DIR / "house_price_pipeline_full.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FORM_SCHEMA_PATH = MODELS_DIR / "form_schema.json"

# Arquivos esperados da distribuição Kaggle
EXPECTED_FILES = ["train.csv", "test.csv", "data_description.txt", "sample_submission.csv"]
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
DICTIONARY_FILE = "data_description.txt"
SUBMISSION_SAMPLE_FILE = "sample_submission.csv"

TARGET = "SalePrice"
ID_COLUMN = "Id"

# Reprodutibilidade e protocolo
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.20
CV_FOLDS = 5
PRIMARY_METRIC = "rmsle"  # definida antes de executar os experimentos (PRD §9.6)

MODEL_VERSION = "2.0.0"
APP_VERSION = "1.0.0"

# Na leitura dos CSVs, apenas o token literal "NA" é tratado como ausente.
# O pandas, por padrão, também converte o texto "None" em NaN, mas "None" é
# uma categoria válida de MasVnrType segundo data_description.txt.
CSV_READ_KWARGS = {"keep_default_na": False, "na_values": ["NA"]}

# Diretórios excluídos da busca pelo ZIP (PRD §6.1)
SEARCH_EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "artifacts"}
