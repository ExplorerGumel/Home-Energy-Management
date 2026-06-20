import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import load_config, ensure_dir
from src.data import HEMSDataLoader, FeatureEngineer
from src.models import ModelFactory, Trainer
from src.evaluation import TimeSeriesEvaluator


def main():
    config = load_config()

    print("[1/7] Loading and validating data...")
    loader = HEMSDataLoader(config)
    df = loader.load()
    report = loader.validate()
    print(f"  Houses: {report['houses']}, Rows: {report['total_rows']:,}")
    print(f"  Date range: {report['date_range'][0]} to {report['date_range'][1]}")

    print("[2/7] Engineering features...")
    fe = FeatureEngineer(config)
    df = fe.build_features(df)
    print(f"  Feature columns after engineering: {len(df.columns)}")

    print("[3/7] Temporal train/val/test split...")
    train_df, val_df, test_df = loader.temporal_split(df)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    print("[4/7] Preparing sequences for deep learning...")
    target_cols = list(config["targets"].values())
    feature_cols = fe.get_feature_columns(df)

    # Save feature columns for API
    ensure_dir("models")
    with open("models/feature_columns.txt", "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    def make_sequences(df_subset, seq_len):
        if len(df_subset) < seq_len:
            return np.empty((0, seq_len, len(feature_cols))), {t: np.empty((0, 1)) for t in target_cols}
        X, y = {t: [] for t in target_cols}, []
        values = df_subset[feature_cols].values
        targets = {t: df_subset[t].values for t in target_cols}
        for i in range(len(values) - seq_len):
            X_seq = values[i : i + seq_len]
            if np.any(np.isnan(X_seq)):
                continue
            X.append(X_seq)
            for t in target_cols:
                y[t].append(targets[t][i + seq_len])
        X_arr = np.array(X)
        y_arr = {t: np.array(y[t]).reshape(-1, 1) for t in target_cols}
        return X_arr, y_arr

    seq_len = config["model"]["sequence_length"]
    X_train, y_train = make_sequences(train_df, seq_len)
    X_val, y_val = make_sequences(val_df, seq_len)
    X_test, y_test = make_sequences(test_df, seq_len)
    n_features = len(feature_cols)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    print("[5/7] Training deep learning model...")
    factory = ModelFactory(config)
    model = factory.build(input_dim=n_features)
    model.summary()

    trainer = Trainer(config)
    model = trainer.fit_keras(model, X_train, y_train, X_val, y_val)

    print("[6/7] Evaluating deep learning model...")
    evaluator = TimeSeriesEvaluator(config)
    preds_all = model.predict(X_test, verbose=0)
    y_pred = {}
    for i, t in enumerate(target_cols):
        y_pred[t] = preds_all[i].flatten()
    y_true = {t: y_test[t].flatten() for t in target_cols}

    results = evaluator.evaluate_all_targets(y_true, y_pred)
    print("\n--- Multi-Task LSTM Results ---")
    evaluator.print_report(results)

    if config["evaluation"]["plot_predictions"]:
        for t in target_cols:
            evaluator.plot_predictions(y_true[t], y_pred[t], t)

    print("[7/7] Training LightGBM baselines...")
    lgb_models = ModelFactory.build_lightgbm_models(config)
    lgb_results = {}

    # Flatten sequences for tabular models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    y_train_vals = np.column_stack([y_train[t] for t in target_cols])
    y_val_vals = np.column_stack([y_val[t] for t in target_cols])
    y_test_vals = np.column_stack([y_test[t] for t in target_cols])

    for i, name in enumerate(target_cols):
        print(f"  Training {name}...")
        model_lgb = lgb_models[name]
        model_lgb.fit(
            X_train_flat,
            y_train_vals[:, i],
            eval_set=[(X_val_flat, y_val_vals[:, i])],
            verbose=False,
        )
        preds = model_lgb.predict(X_test_flat)
        lgb_results[name] = evaluator.compute_all(y_test_vals[:, i], preds)

    print("\n--- LightGBM Results ---")
    evaluator.print_report(lgb_results)

    print("Done!")


if __name__ == "__main__":
    main()
