import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


class HEMSDataLoader:
    def __init__(self, config: dict):
        self.config = config
        data_cfg = config["data"]
        self.raw_path = data_cfg["raw_path"]
        self.timestamp_col = data_cfg["timestamp_col"]
        self.house_id_col = data_cfg["house_id_col"]
        self.date_format = data_cfg["date_format"]
        self.freq = f"{data_cfg['freq_minutes']}min"

        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_path)

        df[self.timestamp_col] = pd.to_datetime(
            df[self.timestamp_col], format=self.date_format
        )
        df = df.sort_values([self.house_id_col, self.timestamp_col]).reset_index(drop=True)

        self.df = df
        return df

    def validate(self) -> dict:
        if self.df is None:
            raise ValueError("Call load() first")

        report = {}
        report["shape"] = self.df.shape
        report["houses"] = self.df[self.house_id_col].nunique()
        report["date_range"] = (
            self.df[self.timestamp_col].min(),
            self.df[self.timestamp_col].max(),
        )
        report["total_rows"] = len(self.df)
        report["nulls"] = self.df.isnull().sum().to_dict()

        # Check expected frequency
        for house_id in self.df[self.house_id_col].unique()[:3]:
            subset = self.df[self.df[self.house_id_col] == house_id]
            diffs = subset[self.timestamp_col].diff().dropna()
            if not diffs.empty:
                report[f"freq_consistency_house_{house_id}"] = {
                    "min_gap": diffs.min(),
                    "max_gap": diffs.max(),
                    "median_gap": diffs.median(),
                }

        return report

    def temporal_split(
        self,
        df: pd.DataFrame,
        train_end: Optional[str] = None,
        val_end: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split_cfg = self.config["training"]["split"]

        train_end = train_end or split_cfg["train_end"]
        val_end = val_end or split_cfg["val_end"]

        train_end_dt = pd.to_datetime(train_end, format=self.date_format)
        val_end_dt = pd.to_datetime(val_end, format=self.date_format)

        train = df[df[self.timestamp_col] <= train_end_dt].copy()
        val = df[
            (df[self.timestamp_col] > train_end_dt)
            & (df[self.timestamp_col] <= val_end_dt)
        ].copy()
        test = df[df[self.timestamp_col] > val_end_dt].copy()

        return train, val, test

    def get_house_split(
        self, df: pd.DataFrame, house_id: int
    ) -> pd.DataFrame:
        return df[df[self.house_id_col] == house_id].copy()

    def get_feature_target_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        target_cols = list(self.config["targets"].values())
        feature_cols = [c for c in df.columns if c not in target_cols]

        X = df[feature_cols].copy()
        y = df[target_cols].copy()

        return X, y
