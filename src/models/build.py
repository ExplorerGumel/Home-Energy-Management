import tensorflow as tf
import lightgbm as lgb
import numpy as np
from typing import Dict, List, Optional


class ModelFactory:
    def __init__(self, config: dict):
        self.config = config
        model_cfg = config["model"]
        self.architecture = model_cfg["architecture"]
        self.seq_len = model_cfg["sequence_length"]
        self.f15_steps = model_cfg["forecast_15_steps"]
        self.f1d_steps = model_cfg["forecast_1d_steps"]

        enc = model_cfg["shared_encoder"]
        self.enc_type = enc["type"]
        self.enc_units = enc["units"]
        self.enc_layers = enc["layers"]
        self.enc_dropout = enc["dropout"]

        self.head_15 = model_cfg["head_15min"]
        self.head_1d = model_cfg["head_1d"]

    def _build_lstm_encoder(self, input_shape: tuple) -> tf.keras.layers.Layer:
        inputs = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
        x = inputs
        for i in range(self.enc_layers):
            return_seq = i < self.enc_layers - 1
            x = tf.keras.layers.LSTM(
                self.enc_units,
                return_sequences=return_seq,
                dropout=self.enc_dropout,
                name=f"lstm_{i}",
            )(x)
        return inputs, x

    def _build_multi_task_model(self, input_dim: int) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.seq_len, input_dim), name="sequence_input")

        if self.enc_type == "lstm":
            x = inputs
            for i in range(self.enc_layers):
                return_seq = i < self.enc_layers - 1
                x = tf.keras.layers.LSTM(
                    self.enc_units,
                    return_sequences=return_seq,
                    dropout=self.enc_dropout,
                    name=f"encoder_lstm_{i}",
                )(x)
        elif self.enc_type == "transformer":
            x = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=self.enc_units // 4
            )(inputs, inputs)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            x = tf.keras.layers.Conv1D(self.enc_units, 3, padding="same", activation="relu")(inputs)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # 15-min head
        h15 = x
        for units in self.head_15["units"]:
            h15 = tf.keras.layers.Dense(units, activation="relu")(h15)
            h15 = tf.keras.layers.Dropout(self.head_15["dropout"])(h15)
        out_15_load = tf.keras.layers.Dense(
            self.f15_steps, name="target_load_kW_15"
        )(h15)
        out_15_pv = tf.keras.layers.Dense(
            self.f15_steps, name="target_pv_kW_15"
        )(h15)

        # 1-day head
        h1d = x
        for units in self.head_1d["units"]:
            h1d = tf.keras.layers.Dense(units, activation="relu")(h1d)
            h1d = tf.keras.layers.Dropout(self.head_1d["dropout"])(h1d)
        out_1d_load = tf.keras.layers.Dense(
            self.f1d_steps, name="target_load_kW_1D"
        )(h1d)
        out_1d_pv = tf.keras.layers.Dense(
            self.f1d_steps, name="target_pv_kW_1D"
        )(h1d)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=[out_15_load, out_15_pv, out_1d_load, out_1d_pv],
        )
        return model

    def build(self, input_dim: int) -> tf.keras.Model:
        if self.architecture == "multi_task":
            return self._build_multi_task_model(input_dim)
        return self._build_multi_task_model(input_dim)

    @staticmethod
    def build_lightgbm_models(config: dict) -> Dict[str, lgb.LGBMRegressor]:
        lgb_params = config["model"]["lightgbm"]
        targets = list(config["targets"].values())

        models = {}
        for target in targets:
            models[target] = lgb.LGBMRegressor(
                num_leaves=lgb_params["num_leaves"],
                learning_rate=lgb_params["learning_rate"],
                n_estimators=lgb_params["n_estimators"],
                early_stopping_rounds=lgb_params["early_stopping_rounds"],
                random_state=42,
                verbosity=-1,
            )
        return models
