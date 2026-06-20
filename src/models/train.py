import os
import numpy as np
import tensorflow as tf
import lightgbm as lgb
from typing import Dict, Optional, Any
from sklearn.multioutput import MultiOutputRegressor


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        train_cfg = config["training"]
        self.batch_size = train_cfg["batch_size"]
        self.epochs = train_cfg["epochs"]
        self.learning_rate = train_cfg["learning_rate"]
        self.optimizer_name = train_cfg["optimizer"]
        self.early_stopping_patience = train_cfg["early_stopping_patience"]
        self.reduce_lr_patience = train_cfg["reduce_lr_patience"]
        self.reduce_lr_factor = train_cfg["reduce_lr_factor"]
        self.model_save_path = train_cfg["model_save_path"]
        self.lgbm_save_path = train_cfg["lgbm_save_path"]
        self.history: Optional[Any] = None

    def _get_optimizer(self):
        opts = {
            "adam": tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
        }
        return opts.get(self.optimizer_name, opts["adam"])

    def _get_callbacks(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=self.reduce_lr_patience,
                factor=self.reduce_lr_factor,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

    def fit_keras(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
    ) -> tf.keras.Model:
        model.compile(
            optimizer=self._get_optimizer(),
            loss={
                "target_load_kW_15": "mse",
                "target_pv_kW_15": "mse",
                "target_load_kW_1D": "mse",
                "target_pv_kW_1D": "mse",
            },
            loss_weights={
                "target_load_kW_15": 1.0,
                "target_pv_kW_15": 1.0,
                "target_load_kW_1D": 0.5,
                "target_pv_kW_1D": 0.5,
            },
        )

        self.history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=self._get_callbacks(),
            verbose=1,
        )

        model.load_weights(self.model_save_path)
        return model

    def fit_lightgbm(
        self,
        models: Dict[str, lgb.LGBMRegressor],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_names: list,
    ) -> Dict[str, lgb.LGBMRegressor]:
        trained = {}
        for name in target_names:
            col_idx = target_names.index(name)
            model = models[name]
            model.fit(
                X_train,
                y_train[:, col_idx],
                eval_set=[(X_val, y_val[:, col_idx])],
                verbose=50,
            )
            trained[name] = model
        return trained

    def save_lightgbm(self, models: Dict[str, lgb.LGBMRegressor]):
        os.makedirs(os.path.dirname(self.lgbm_save_path), exist_ok=True)
        for name, model in models.items():
            path = self.lgbm_save_path.replace(".txt", f"_{name}.txt")
            model.booster_.save_model(path)
