import numpy as np
import pandas as pd
from typing import List, Optional


class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        fe_cfg = config["feature_engineering"]
        self.cyclical_cols = fe_cfg["cyclical_encode"]
        self.lag_cfg = fe_cfg["lag_features"]
        self.rolling_cfg = fe_cfg["rolling_features"]
        self.appliance_cfg = fe_cfg["appliance_features"]
        self.freq_minutes = config["data"]["freq_minutes"]

    def cyclical_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mapping = {
            "hour_of_day": (24, 360),
            "minute_of_hour": (60, 360),
            "day_of_week": (7, 360),
        }
        for col in self.cyclical_cols:
            if col not in df.columns:
                continue
            max_val, divisor = mapping.get(col, (24, 360))
            angle = 2 * np.pi * df[col] / max_val
            df[f"{col}_sin"] = np.sin(angle)
            df[f"{col}_cos"] = np.cos(angle)
        return df

    def create_lags(self, df: pd.DataFrame, horizon: str = "15min") -> pd.DataFrame:
        if not self.lag_cfg["enabled"]:
            return df

        df = df.copy()
        columns = self.lag_cfg["columns"]

        if horizon == "15min":
            lags = self.lag_cfg["lags_15min"]
        else:
            lags = self.lag_cfg["lags_1d"]

        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df.groupby(self.config["data"]["house_id_col"])[col].shift(lag)

        return df

    def create_rolling(self, df: pd.DataFrame, horizon: str = "15min") -> pd.DataFrame:
        if not self.rolling_cfg["enabled"]:
            return df

        df = df.copy()
        columns = self.rolling_cfg["columns"]

        if horizon == "15min":
            windows = self.rolling_cfg["windows_15min"]
        else:
            windows = self.rolling_cfg["windows_1d"]

        for col in columns:
            if col not in df.columns:
                continue
            for w in windows:
                grouped = df.groupby(self.config["data"]["house_id_col"])[col]
                df[f"{col}_rolling_mean_{w}"] = grouped.transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                )
                df[f"{col}_rolling_std_{w}"] = grouped.transform(
                    lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
                )
        return df

    def create_appliance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.appliance_cfg["enabled"]:
            return df

        df = df.copy()
        appliances = ["washing_machine_on", "dishwasher_on", "water_heater_on", "ev_charging_on"]
        window = self.appliance_cfg["time_since_window"]

        for app in appliances:
            if app not in df.columns:
                continue
            # Time since last ON event (in steps)
            col_name = f"{app}_steps_since_on"
            df[col_name] = df.groupby(self.config["data"]["house_id_col"])[app].transform(
                lambda x: self._steps_since_event(x, window)
            )

            # Rolling count of ON events in window
            count_name = f"{app}_count_{window}"
            df[count_name] = df.groupby(self.config["data"]["house_id_col"])[app].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum().fillna(0)
            )

        return df

    def _steps_since_event(self, series: pd.Series, max_steps: int) -> pd.Series:
        result = np.full(len(series), max_steps, dtype=float)
        last_on = -1
        for i in range(len(series)):
            if series.iloc[i] == 1:
                last_on = i
            if last_on >= 0:
                result[i] = min(i - last_on, max_steps)
        return result

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.cyclical_encode(df)
        df = self.create_lags(df, horizon="15min")
        df = self.create_lags(df, horizon="1d")
        df = self.create_rolling(df, horizon="15min")
        df = self.create_rolling(df, horizon="1d")
        df = self.create_appliance_features(df)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        dropped = list(self.config["targets"].values()) + [
            self.config["data"]["timestamp_col"],
            self.config["data"]["house_id_col"],
        ]
        return [c for c in df.columns if c not in dropped]
