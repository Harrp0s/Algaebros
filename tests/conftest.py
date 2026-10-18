"""Test setup: every test run uses a throwaway data folder and SYNTHETIC data (never ./data)."""
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["ALGAEBRO_DATA_DIR"] = tempfile.mkdtemp(prefix="algaebro_test_")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import pytest


@pytest.fixture(scope="session")
def synth():
    """Generate synthetic data once, train on it, and return the data folder."""
    import make_synthetic_data as gen
    from algaebro import config, features, model, weather
    import pandas as pd

    gen.main(weather_effect=1.0, seed=1, end=date(2026, 10, 15))   # pinned: the tests must not depend on today's date
    obs = features.prepare_obs(pd.read_csv(config.NDCI_CSV))
    wfeat = features.weather_features(weather.load())
    table = features.build_training_table(obs, wfeat)
    model.train_and_save(table)
    return config.DATA_DIR


@pytest.fixture()
def store(tmp_path):
    from algaebro.store import Store

    return Store(tmp_path / "test.db")


@pytest.fixture(scope="session")
def replay_fc(synth):
    """A forecast as of a hot July day, which has HIGH zones in this synthetic data."""
    from algaebro import forecast

    return forecast.make_forecast("2026-07-20")
