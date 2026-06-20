# Home Energy Management System — Multi-Horizon Forecasting

Forecast home energy load and PV generation at 15-minute and 1-day horizons using deep learning and gradient boosting.

## Problem

Given historical smart meter data (weather, appliance states, battery status, grid interaction), predict:
- `target_load_kW_15` — load 15 minutes ahead
- `target_pv_kW_15` — PV generation 15 minutes ahead
- `target_load_kW_1D` — load 24 hours ahead
- `target_pv_kW_1D` — PV generation 24 hours ahead

## Architecture

### Multi-Task Deep Learning

```
Input (96 timesteps × N features)
  │
  ▼
LSTM Encoder (2 layers, 128 units)
  │
  ├──→ Dense Head (15-min) ──→ load_15, pv_15
  │
  └──→ Dense Head (1-day) ───→ load_1d, pv_1d
```

### LightGBM Baseline
Separate regressor per target using flattened sequence input.

## Project Structure

```
├── config/config.yaml              # All hyperparameters
├── src/
│   ├── data/
│   │   ├── loader.py               # Load, validate, temporal split
│   │   └── features.py             # Cyclical, lags, rolling, appliance features
│   ├── models/
│   │   ├── build.py                # Multi-task LSTM + LightGBM
│   │   └── train.py                # Keras + LightGBM training pipelines
│   ├── evaluation/metrics.py       # MAE, RMSE, MAPE, R² per target
│   └── api/serve.py                # FastAPI inference endpoint
├── notebooks/01-EDA.ipynb          # Exploratory analysis
├── notebooks/02-Modeling.ipynb     # Training & evaluation
├── run_pipeline.py                 # Full pipeline runner
├── Dockerfile
└── requirements.txt
```

## Setup & Run

```bash
pip install -r requirements.txt

# Download HEMS_dataset.csv to data/raw/

# Run full pipeline
python run_pipeline.py

# Start inference API
python -m src.api.serve
```

## Feature Engineering

| Group | Features |
|-------|----------|
| Cyclical time | hour_of_day, day_of_week, minute_of_hour (sin/cos) |
| Lags (15-min) | demand, PV, temp, irradiance, battery at t-1..t-24 |
| Lags (1-day) | demand, PV at t-96 |
| Rolling stats | mean/std over 1h, 2h, 6h, 24h, 3.5d windows |
| Appliance | steps_since_on, rolling ON count for washer, dishwasher, heater, EV |

## Evaluation

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |

Temporal walk-forward split: train → val → test (no leakage).

## Deployment

```bash
# Docker
docker build -t hems-api .
docker run -p 8000:8000 hems-api

# API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.36, 0.18, ...]}'
```

## License

MIT
