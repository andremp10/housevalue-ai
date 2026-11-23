"""Testes de dados (PRD §18.1): estrutura calculada a partir dos arquivos, não presumida."""
import json

import pandas as pd

from ml import config
from ml.data import parse_data_dictionary, sha256_file, structural_na_columns


def test_working_copy_matches_manifest():
    manifest = json.loads(config.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    for item in manifest["working_copy"]:
        assert sha256_file(config.PROJECT_ROOT / item["path"]) == item["sha256"]
    if manifest["zip"]:
        names = {m["name"] for m in manifest["zip_members"]}
        assert set(config.EXPECTED_FILES) <= names


def test_shapes_and_target(train_df, test_df):
    assert config.TARGET in train_df.columns
    assert config.TARGET not in test_df.columns
    assert pd.api.types.is_numeric_dtype(train_df[config.TARGET])
    assert train_df[config.TARGET].isna().sum() == 0
    assert (train_df[config.TARGET] > 0).all()
    assert train_df.shape[1] == test_df.shape[1] + 1
    profile = json.loads(config.DATA_PROFILE_PATH.read_text(encoding="utf-8"))
    assert profile["files"]["train.csv"]["rows"] == len(train_df)
    assert profile["files"]["test.csv"]["rows"] == len(test_df)


def test_ids_unique_and_disjoint(train_df, test_df):
    assert train_df[config.ID_COLUMN].is_unique and test_df[config.ID_COLUMN].is_unique
    assert set(train_df[config.ID_COLUMN]).isdisjoint(test_df[config.ID_COLUMN])


def test_none_is_a_category_not_missing(train_df):
    # "None" é categoria válida de MasVnrType no dicionário; não pode virar NaN na leitura.
    assert (train_df["MasVnrType"] == "None").sum() > 0
    assert train_df["MasVnrType"].isna().sum() < (train_df["MasVnrType"] == "None").sum()


def test_dictionary_covers_all_features(train_df):
    d = parse_data_dictionary()
    feats = [c for c in train_df.columns if c not in (config.ID_COLUMN, config.TARGET)]
    assert all(c in d for c in feats)
    structural = structural_na_columns(d)
    assert {"PoolQC", "GarageType", "BsmtQual", "Alley", "Fence"} <= set(structural)
    assert "LotFrontage" not in structural
