import pytest

from ml import config


def _require(*paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Artefatos ausentes ({', '.join(p.name for p in missing)}). Rode ml.inspect_data / ml.train / ml.evaluate.")


@pytest.fixture(scope="session")
def train_df():
    _require(config.RAW_DIR / config.TRAIN_FILE)
    from ml.data import load_train
    return load_train()


@pytest.fixture(scope="session")
def test_df():
    _require(config.RAW_DIR / config.TEST_FILE)
    from ml.data import load_test
    return load_test()


@pytest.fixture(scope="session")
def trained():
    _require(config.APP_MODEL_PATH, config.MODEL_METADATA_PATH, config.SPLIT_PATH, config.HOLDOUT_REPORT_PATH)
    return True
