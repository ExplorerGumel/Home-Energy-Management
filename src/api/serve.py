import os
import sys
import numpy as np
import tensorflow as tf
import lightgbm as lgb
from typing import Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.helpers import load_config

app = FastAPI(title="HEMS Forecast API", version="1.0.0")
config = load_config()
model = None
lgb_models = {}
targets = list(config["targets"].values())
feature_cols: List[str] = []


class InferenceInput(BaseModel):
    features: List[float]


class InferenceOutput(BaseModel):
    predictions: Dict[str, float]


@app.on_event("startup")
def load_model():
    global model, lgb_models, feature_cols

    # Load feature columns from training metadata
    meta_path = "models/feature_columns.txt"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]

    # Try loading Keras model
    keras_path = config["training"]["model_save_path"]
    if os.path.exists(keras_path):
        global model
        model = tf.keras.models.load_model(keras_path)

    # Try loading LightGBM models
    lgbm_path = config["training"]["lgbm_save_path"]
    for target in targets:
        path = lgbm_path.replace(".txt", f"_{target}.txt")
        if os.path.exists(path):
            lgb_models[target] = lgb.Booster(model_file=path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None or len(lgb_models) > 0}


@app.post("/predict", response_model=InferenceOutput)
def predict(input_data: InferenceInput):
    features = np.array(input_data.features).reshape(1, -1)
    predictions = {}

    if model is not None:
        preds = model.predict(features, verbose=0)
        names = list(config["targets"].values())
        for i, name in enumerate(names):
            predictions[name] = float(preds[i].flatten()[0])
    elif lgb_models:
        for name, lgb_model in lgb_models.items():
            predictions[name] = float(lgb_model.predict(features)[0])
    else:
        raise RuntimeError("No model loaded")

    return InferenceOutput(predictions=predictions)


def main():
    api_cfg = config["api"]
    uvicorn.run(
        "src.api.serve:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
