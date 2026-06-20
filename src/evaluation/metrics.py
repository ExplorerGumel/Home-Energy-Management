import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class TimeSeriesEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def compute_all(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        return {
            "mae": self.mae(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
        }

    def evaluate_all_targets(
        self,
        y_true_dict: Dict[str, np.ndarray],
        y_pred_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for name in y_true_dict:
            results[name] = self.compute_all(
                y_true_dict[name], y_pred_dict[name]
            )
        return results

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
        timestamps: Optional[List[str]] = None,
        n_samples: int = 500,
    ):
        fig, ax = plt.subplots(figsize=(14, 5))
        n = min(n_samples, len(y_true))
        idx = np.arange(n)

        ax.plot(idx, y_true[:n], label="Actual", alpha=0.8)
        ax.plot(idx, y_pred[:n], label="Predicted", alpha=0.8, linestyle="--")
        ax.set_title(f"{target_name} — Actual vs Predicted (first {n} samples)")
        ax.set_xlabel("Sample")
        ax.set_ylabel(target_name)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def print_report(
        self, results: Dict[str, Dict[str, float]]
    ) -> str:
        lines = []
        lines.append(f"{'Target':<25} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R2':<10}")
        lines.append("-" * 65)
        for target, metrics in results.items():
            lines.append(
                f"{target:<25} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
                f"{metrics['mape']:<10.2f} {metrics['r2']:<10.4f}"
            )
        report = "\n".join(lines)
        print(report)
        return report
