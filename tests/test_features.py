import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.features import FeatureEngineer


@pytest.fixture
def config():
    return {
        "data": {"house_id_col": "house_id", "timestamp_col": "timestamp", "freq_minutes": 15, "date_format": "%d-%m-%Y %H:%M"},
        "feature_engineering": {
            "cyclical_encode": ["hour_of_day", "day_of_week", "minute_of_hour"],
            "lag_features": {
                "enabled": True,
                "columns": ["total_demand_kW"],
                "lags_15min": [1, 4],
                "lags_1d": [96],
            },
            "rolling_features": {
                "enabled": True,
                "windows_15min": [4],
                "windows_1d": [96],
                "columns": ["total_demand_kW"],
            },
            "appliance_features": {
                "enabled": True,
                "time_since_window": 96,
            },
        },
    }


def test_cyclical_encode(config):
    df = pd.DataFrame({"hour_of_day": [0, 6, 12, 18], "day_of_week": [0, 3, 6, 1], "minute_of_hour": [0, 15, 30, 45]})
    fe = FeatureEngineer(config)
    result = fe.cyclical_encode(df)
    assert "hour_of_day_sin" in result.columns
    assert "hour_of_day_cos" in result.columns
    assert "day_of_week_sin" in result.columns
    assert abs(result["hour_of_day_sin"].iloc[0]) < 1e-10


def test_create_lags(config):
    df = pd.DataFrame({
        "house_id": [1] * 100,
        "total_demand_kW": np.random.randn(100),
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="15min"),
    })
    fe = FeatureEngineer(config)
    result = fe.create_lags(df, horizon="15min")
    assert "total_demand_kW_lag_1" in result.columns
    assert "total_demand_kW_lag_4" in result.columns
    assert pd.isna(result["total_demand_kW_lag_1"].iloc[0])


def test_create_rolling(config):
    df = pd.DataFrame({
        "house_id": [1] * 100,
        "total_demand_kW": np.random.randn(100),
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="15min"),
    })
    fe = FeatureEngineer(config)
    result = fe.create_rolling(df, horizon="15min")
    assert "total_demand_kW_rolling_mean_4" in result.columns
