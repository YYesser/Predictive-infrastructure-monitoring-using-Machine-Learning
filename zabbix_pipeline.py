from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Dict, Optional, Sequence

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "zabbix_data.csv"
HOSTS_DIR = BASE_DIR / "data" / "hosts"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
CHUNK_SIZE = 50000
RECENT_WINDOW_SIZE = 2000
DEFAULT_CONFIDENCE_THRESHOLD = 0.15


def validate_backtest_confidence(training_summary_path: str = "reports/training_summary.csv") -> bool:
    """Validate that backtest_confidence behaves sensibly on exported training data."""
    summary_path = Path(training_summary_path)
    if not summary_path.exists():
        print(f"VALIDATION FAILED: training summary not found at {summary_path}")
        return False

    df = pd.read_csv(summary_path)
    issues: List[str] = []

    null_conf = int(df["backtest_confidence"].isna().sum())
    if null_conf > 0:
        issues.append(f"{null_conf} rows have null backtest_confidence")

    out_of_range = df[(df["backtest_confidence"] < 0) | (df["backtest_confidence"] > 1)]
    if not out_of_range.empty:
        issues.append(f"{len(out_of_range)} rows have backtest_confidence outside [0,1]")

    dead_series = df[df["selection_reason"].astype(str).str.contains("nearly constant", na=False)]
    high_conf_dead = dead_series[dead_series["backtest_confidence"] > 0.5]
    if not high_conf_dead.empty:
        issues.append(
            f"{len(high_conf_dead)} rows flagged as 'nearly constant / no signal' still show backtest_confidence > 0.5 - scale bug may still be present:\n"
            f"{high_conf_dead[['host_name', 'item_key', 'health_score', 'backtest_confidence']].to_string()}"
        )

    volatile_series = df[df["selection_reason"].astype(str).str.contains("volatility was too high", na=False)]
    if not volatile_series.empty:
        winning_mape_cols = {
            "baseline": "baseline_mape",
            "trend_baseline": "trend_mape",
            "rolling_baseline": "rolling_mape",
        }
        winning_mape_values = []
        for _, row in volatile_series.iterrows():
            strategy = str(row.get("forecast_strategy", "baseline") or "baseline")
            mape_col = winning_mape_cols.get(strategy, "baseline_mape")
            mape_value = row.get(mape_col)
            winning_mape_values.append(float(mape_value) if pd.notna(mape_value) else float("nan"))

        volatile_series = volatile_series.copy()
        volatile_series["winning_strategy_mape"] = winning_mape_values
        suspicious_volatile = volatile_series[
            (volatile_series["health_score"] <= 0.8)
            & (volatile_series["backtest_confidence"] > 0.8)
            & (volatile_series["winning_strategy_mape"] > 50.0)
        ]
        if not suspicious_volatile.empty:
            issues.append(
                f"{len(suspicious_volatile)} high-volatility-rejected rows still show confidence > 0.8 despite poor fallback MAPE:\n"
                f"{suspicious_volatile[['host_name', 'item_key', 'health_score', 'backtest_confidence', 'winning_strategy_mape']].to_string()}"
            )

    numeric_mask = df["mae"].notna() & df["backtest_confidence"].notna() & (df["mae"] > 0)
    if numeric_mask.sum() > 10:
        log_mae = np.log1p(df.loc[numeric_mask, "mae"])
        corr = np.corrcoef(log_mae, df.loc[numeric_mask, "backtest_confidence"])[0, 1]
        if abs(corr) > 0.5:
            issues.append(
                f"backtest_confidence correlates with log(mae) at r={corr:.2f} - this suggests confidence is still scale-dependent rather than relative-error based"
            )

    if issues:
        print("VALIDATION FAILED:")
        for issue in issues:
            print(f" - {issue}")
        return False

    print(
        f"VALIDATION PASSED - {len(df)} rows checked, backtest_confidence looks scale-invariant and correctly bounded."
    )
    return True


def sanitize_name(value: str) -> str:
    """Convert host/item names into safe filenames."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return sanitized.strip("_") or "unknown"


def _normalize_column_name(column_name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column_name).strip().lower()).strip("_")


def _find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    lookup = {_normalize_column_name(col): col for col in columns}
    for alias in aliases:
        canonical = _normalize_column_name(alias)
        if canonical in lookup:
            return lookup[canonical]
    return None


def normalize_metric_frame(
    frame: pd.DataFrame,
    host_fallback: str = "unknown",
    item_fallback: str = "unknown",
    max_expected_span_days: int = 90,
) -> pd.DataFrame:
    """Normalize common CSV schemas so uploaded metrics can be used by the training pipeline."""
    normalized = frame.copy()
    normalized.columns = [str(col) for col in normalized.columns]

    host_col = _find_column(normalized.columns, ["host_name", "host", "hostname", "server", "device"])
    item_key_col = _find_column(normalized.columns, ["item_key", "metric_key", "metric", "metric_name", "item", "series", "name"])
    item_name_col = _find_column(normalized.columns, ["item_name", "metric_name", "description", "label", "series_name"])
    timestamp_col = _find_column(normalized.columns, ["timestamp", "ts", "time", "datetime", "date", "date_time", "event_time"])
    value_col = _find_column(normalized.columns, ["value", "metric_value", "measurement", "val", "reading", "data", "metric"])
    value_type_col = _find_column(normalized.columns, ["value_type", "type", "valuetype", "kind"])

    if host_col is not None:
        host_values = normalized[host_col].fillna(host_fallback)
    else:
        host_values = pd.Series([host_fallback] * len(normalized), index=normalized.index, dtype=str)
    normalized["host_name"] = host_values.astype(str).replace({"nan": host_fallback})

    if item_key_col is not None:
        item_key_values = normalized[item_key_col].fillna(item_fallback)
    elif item_name_col is not None:
        item_key_values = normalized[item_name_col].fillna(item_fallback)
    else:
        item_key_values = pd.Series([item_fallback] * len(normalized), index=normalized.index, dtype=str)

    if item_name_col is not None:
        item_name_values = normalized[item_name_col].fillna(item_key_values)
    else:
        item_name_values = item_key_values

    normalized["item_key"] = item_key_values.astype(str).replace({"nan": item_fallback})
    normalized["item_name"] = item_name_values.astype(str).replace({"nan": item_fallback})

    # Parse timestamps when a column is available; otherwise leave as NaT so we can
    # fill missing timestamps per host/item group (avoids assigning a single
    # global date range across the entire uploaded file).
    if timestamp_col is not None:
        normalized["timestamp"] = pd.to_datetime(normalized[timestamp_col], errors="coerce")
    else:
        normalized["timestamp"] = pd.NaT

    # Remember original order so group-based filling doesn't reorder rows.
    normalized["_orig_order"] = np.arange(len(normalized))

    # Fill timestamps per (host_name, item_key) group.
    if normalized["timestamp"].isna().any():
        missing_total = int(normalized["timestamp"].isna().sum())
        total = len(normalized)
        if missing_total == total:
            logging.warning(
                f"normalize_metric_frame: all {total} rows are missing timestamps; "
                "filling per-host/item with synthetic hourly sequences starting at 2024-01-01"
            )
        else:
            logging.warning(
                f"normalize_metric_frame: {missing_total}/{total} rows missing timestamps; "
                "filling missing timestamps per host/item by forward/back-fill or per-group sequences"
            )
        filled_frames: list[pd.DataFrame] = []
        for (host_val, item_val), g in normalized.groupby(["host_name", "item_key"], sort=False):
            g2 = g.copy()
            # Ensure the group keeps these key columns
            g2["host_name"] = host_val
            g2["item_key"] = item_val

            # If the entire series for this host/item has no timestamps,
            # assign a simple hourly sequence anchored at a fixed base date.
            if g2["timestamp"].isna().all():
                g2["timestamp"] = pd.date_range(start="2024-01-01", periods=len(g2), freq="h")
                filled_frames.append(g2)
                continue

            # If some timestamps exist, forward/back-fill the missing ones.
            if g2["timestamp"].isna().any():
                g2["timestamp"] = g2["timestamp"].ffill().bfill()

            # Any remaining NaT (unlikely) — fill with an hourly sequence
            # starting from the group's minimum timestamp or a fixed base.
            if g2["timestamp"].isna().any():
                try:
                    base = g2["timestamp"].min()
                    if pd.isna(base):
                        base = pd.Timestamp("2024-01-01")
                except Exception:
                    base = pd.Timestamp("2024-01-01")
                missing_idx = g2["timestamp"].isna()
                seq = pd.date_range(start=base, periods=missing_idx.sum(), freq="h")
                g2.loc[missing_idx, "timestamp"] = seq

            filled_frames.append(g2)

        if filled_frames:
            normalized = pd.concat(filled_frames, ignore_index=True)
            # restore original order and drop helper column
            normalized = normalized.sort_values("_orig_order").reset_index(drop=True)

    normalized = normalized.drop(columns=["_orig_order"], errors="ignore")

    if value_col is not None:
        normalized["value"] = pd.to_numeric(normalized[value_col], errors="coerce")
    else:
        numeric_candidates = [col for col in normalized.columns if col not in {"host_name", "item_name", "item_key", "timestamp", "value", "value_type"}]
        normalized["value"] = pd.Series([np.nan] * len(normalized), index=normalized.index, dtype=float)
        for candidate in numeric_candidates:
            numeric_values = pd.to_numeric(normalized[candidate], errors="coerce")
            if numeric_values.notna().any():
                normalized["value"] = numeric_values
                break

    if value_type_col is not None:
        normalized["value_type"] = pd.to_numeric(normalized[value_type_col], errors="coerce").fillna(0)
    else:
        normalized["value_type"] = 0
    normalized["value_type"] = normalized["value_type"].fillna(0).astype(int)

    normalized = normalized.dropna(subset=["timestamp", "value"]).copy()

    # Sanity check: warn if any host/item group's timestamp span exceeds a reasonable threshold.
    try:
        spans = (
            normalized.groupby(["host_name", "item_key"], sort=False)["timestamp"]
            .agg(["min", "max"])
            .reset_index()
        )
        spans["span_days"] = (spans["max"] - spans["min"]).dt.total_seconds() / 86400.0
        large_spans = spans[spans["span_days"] > float(max_expected_span_days)]
        if not large_spans.empty:
            for _, row in large_spans.iterrows():
                try:
                    logging.warning(
                        "Large timestamp span detected for host=%s item_key=%s: span_days=%.2f",
                        row["host_name"],
                        row["item_key"],
                        float(row["span_days"]),
                    )
                except Exception:
                    logging.warning(
                        "Large timestamp span detected for a host/item (could not format details)."
                    )
    except Exception:
        # Avoid failing normalization if span check encounters unexpected data types.
        logging.exception("Failed to compute per-group timestamp spans for sanity check")

    return normalized[["host_name", "item_name", "item_key", "timestamp", "value", "value_type"]]


def build_feature_columns() -> List[str]:
    """Return the ordered feature columns used by the forecasting models."""
    return [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_5",
        "rolling_std_5",
        "rolling_mean_10",
        "rolling_std_10",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "zscore_recent",
    ]


def engineer_feature_frame(group: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and temporal features from the historical series."""
    engineered = group.copy()
    engineered = engineered.sort_values("timestamp").reset_index(drop=True)
    engineered["lag_1"] = engineered["value"].shift(1)
    engineered["lag_2"] = engineered["value"].shift(2)
    engineered["lag_3"] = engineered["value"].shift(3)
    engineered["rolling_mean_3"] = engineered["value"].shift(1).rolling(window=3, min_periods=3).mean()
    engineered["rolling_std_3"] = engineered["value"].shift(1).rolling(window=3, min_periods=3).std()
    engineered["rolling_mean_5"] = engineered["value"].shift(1).rolling(window=5, min_periods=5).mean()
    engineered["rolling_std_5"] = engineered["value"].shift(1).rolling(window=5, min_periods=5).std()
    engineered["rolling_mean_10"] = engineered["value"].shift(1).rolling(window=10, min_periods=10).mean()
    engineered["rolling_std_10"] = engineered["value"].shift(1).rolling(window=10, min_periods=10).std()
    engineered["zscore_recent"] = (
        (engineered["value"] - engineered["rolling_mean_10"]) / engineered["rolling_std_10"].replace(0, np.nan)
    ).fillna(0.0)

    ts = engineered["timestamp"]
    engineered["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    engineered["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    engineered["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    engineered["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    engineered["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12.0)
    engineered["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12.0)
    return engineered


def build_feature_vector_from_history(history: Sequence[float] | pd.Series, timestamp: Optional[pd.Timestamp] = None) -> List[float]:
    """Build the same feature vector used during training from the latest observed history."""
    values = pd.Series(history, dtype=float).astype(float)
    if values.empty:
        return [0.0] * len(build_feature_columns())

    values = values.dropna().reset_index(drop=True)
    if values.empty:
        return [0.0] * len(build_feature_columns())

    recent = values.tail(10).to_numpy(dtype=float)
    lag_1 = float(recent[-1])
    lag_2 = float(recent[-2]) if len(recent) >= 2 else lag_1
    lag_3 = float(recent[-3]) if len(recent) >= 3 else lag_1
    rolling_mean_3 = float(np.mean(recent[-3:])) if len(recent) >= 3 else float(np.mean(recent))
    rolling_std_3 = float(np.std(recent[-3:])) if len(recent) >= 3 else float(np.std(recent))
    rolling_mean_5 = float(np.mean(recent[-5:])) if len(recent) >= 5 else float(np.mean(recent))
    rolling_std_5 = float(np.std(recent[-5:])) if len(recent) >= 5 else float(np.std(recent))
    rolling_mean_10 = float(np.mean(recent))
    rolling_std_10 = float(np.std(recent))
    zscore_recent = 0.0 if rolling_std_10 == 0 else (lag_1 - rolling_mean_10) / rolling_std_10

    if timestamp is None:
        timestamp = pd.Timestamp.utcnow()

    hour_sin = float(np.sin(2 * np.pi * timestamp.hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * timestamp.hour / 24.0))
    dow_sin = float(np.sin(2 * np.pi * timestamp.dayofweek / 7.0))
    dow_cos = float(np.cos(2 * np.pi * timestamp.dayofweek / 7.0))
    month_sin = float(np.sin(2 * np.pi * timestamp.month / 12.0))
    month_cos = float(np.cos(2 * np.pi * timestamp.month / 12.0))

    return [
        lag_1,
        lag_2,
        lag_3,
        rolling_mean_3,
        rolling_std_3,
        rolling_mean_5,
        rolling_std_5,
        rolling_mean_10,
        rolling_std_10,
        hour_sin,
        hour_cos,
        dow_sin,
        dow_cos,
        month_sin,
        month_cos,
        zscore_recent,
    ]


def generate_simple_forecast(values: Sequence[float], days: int = 7) -> List[float]:
    """Create a simple 7-day persistence forecast from any numeric history."""
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return [0.0] * days
    return [cleaned[-1]] * days


def generate_trend_forecast(values: Sequence[float], days: int = 7) -> List[float]:
    """Create a 7-day forecast using the recent trend in the series."""
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return [0.0] * days
    if len(cleaned) == 1:
        return [cleaned[-1]] * days

    recent = cleaned[-min(10, len(cleaned)):]
    x = np.arange(len(recent), dtype=float)
    y = np.asarray(recent, dtype=float)
    slope = 0.0
    if len(recent) > 1:
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            slope = float(np.mean(np.diff(recent))) if len(recent) > 1 else 0.0

    if not np.isfinite(slope):
        slope = 0.0

    current = float(recent[-1])
    forecast: List[float] = []
    for _ in range(days):
        current = current + slope
        forecast.append(current)
    return forecast


def generate_rolling_mean_forecast(values: Sequence[float], days: int = 7, window: int = 3) -> List[float]:
    """Create a 7-day forecast using the rolling mean of the recent series."""
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return [0.0] * days
    if len(cleaned) == 1:
        return [cleaned[-1]] * days

    recent = cleaned[-min(window, len(cleaned)):]
    baseline = float(np.mean(recent))
    return [baseline] * days


def compute_error_metrics(actuals: Sequence[float] | np.ndarray, predictions: Sequence[float] | np.ndarray) -> Dict[str, float | str]:
    """Return MAE, RMSE, MAPE, and a status flag for a set of actual versus predicted values."""
    actual_array = np.asarray(actuals, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    mae = float(mean_absolute_error(actual_array, prediction_array))
    rmse = float(np.sqrt(mean_squared_error(actual_array, prediction_array)))

    non_zero = np.abs(actual_array) > 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((actual_array[non_zero] - prediction_array[non_zero]) / actual_array[non_zero])) * 100)
        mape_status = "ok"
    else:
        mape = float("nan")
        mape_status = "undefined_zero_actuals"

    return {"mae": mae, "rmse": rmse, "mape": mape, "mape_status": mape_status}


def evaluate_series_health(values: Sequence[float] | np.ndarray | pd.Series, min_length: int = 30) -> Dict[str, object]:
    """Assess whether a series is healthy enough for a trained-model strategy to be trusted."""
    cleaned = [float(value) for value in values if pd.notna(value)]
    if not cleaned:
        return {
            "is_healthy": False,
            "health_score": 0.0,
            "health_status": "empty",
            "health_reason": "series was empty",
            "series_length": 0,
            "mean": float("nan"),
            "std": float("nan"),
        }

    series = np.asarray(cleaned, dtype=float)
    series_length = int(len(series))
    mean = float(np.mean(series))
    std = float(np.std(series, ddof=0))
    baseline = max(abs(mean), 1e-6)
    coefficient_of_variation = std / baseline if baseline > 1e-6 else float("inf")

    if series_length < min_length:
        return {
            "is_healthy": False,
            "health_score": 0.0,
            "health_status": "too_short",
            "health_reason": f"series length {series_length} is below the minimum {min_length}",
            "series_length": series_length,
            "mean": mean,
            "std": std,
            "coefficient_of_variation": coefficient_of_variation,
        }

    if std <= 1e-8:
        return {
            "is_healthy": False,
            "health_score": 0.0,
            "health_status": "near_constant",
            "health_reason": "series is nearly constant and does not carry enough signal",
            "series_length": series_length,
            "mean": mean,
            "std": std,
            "coefficient_of_variation": coefficient_of_variation,
        }

    signal_strength = min(1.0, max(0.0, (std / baseline) / 0.2))
    length_score = min(1.0, series_length / max(min_length, 60))
    variance_score = min(1.0, max(0.0, 1.0 - min(1.0, abs(coefficient_of_variation) / 3.0)))
    health_score = float(np.clip(0.5 * signal_strength + 0.3 * length_score + 0.2 * variance_score, 0.0, 1.0))
    is_healthy = health_score >= 0.25 and std > 1e-8
    health_status = "healthy" if is_healthy else "weak_signal"
    health_reason = "series has enough signal for model-based forecasting" if is_healthy else "series has weak signal and should prefer a simple baseline"

    return {
        "is_healthy": bool(is_healthy),
        "health_score": health_score,
        "health_status": health_status,
        "health_reason": health_reason,
        "series_length": series_length,
        "mean": mean,
        "std": std,
        "coefficient_of_variation": coefficient_of_variation,
    }


def select_validation_predictions(
    forecast_strategy: str,
    *,
    model_predictions: Optional[Sequence[float] | np.ndarray] = None,
    baseline_predictions: Optional[Sequence[float] | np.ndarray] = None,
    trend_predictions: Optional[Sequence[float] | np.ndarray] = None,
    rolling_predictions: Optional[Sequence[float] | np.ndarray] = None,
) -> List[float]:
    """Return the validation predictions that match the selected forecast strategy."""
    strategy = str(forecast_strategy or "").strip().lower()
    if strategy == "trend_baseline" and trend_predictions is not None:
        return [float(value) for value in trend_predictions]
    if strategy == "rolling_baseline" and rolling_predictions is not None:
        return [float(value) for value in rolling_predictions]
    if strategy == "baseline" and baseline_predictions is not None:
        return [float(value) for value in baseline_predictions]
    if model_predictions is not None:
        return [float(value) for value in model_predictions]
    return []


def save_validation_predictions_csv(
    host_name: str,
    item_key: str,
    timestamps: Sequence[pd.Timestamp] | np.ndarray,
    actuals: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | np.ndarray,
    output_dir: Path,
) -> Path:
    """Save actual and predicted validation values to a CSV file without plotting them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_array = np.asarray(actuals, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    timestamp_array = [pd.Timestamp(ts).isoformat() if not pd.isna(ts) else "" for ts in timestamps]
    errors = np.abs(actual_array - prediction_array)
    frame = pd.DataFrame(
        {
            "host_name": [host_name] * len(actual_array),
            "item_key": [item_key] * len(actual_array),
            "timestamp": timestamp_array,
            "actual_value": actual_array,
            "predicted_value": prediction_array,
            "abs_error": errors,
            "squared_error": np.square(errors),
        }
    )
    output_path = output_dir / f"{sanitize_name(host_name)}_{sanitize_name(str(item_key))}_validation.csv"
    frame.to_csv(output_path, index=False)
    return output_path


def _compute_backtest_confidence(
    model_mae: float,
    baseline_mae: float,
    trend_mae: Optional[float] = None,
    rolling_mae: Optional[float] = None,
    health: Optional[Dict[str, object]] = None,
    winning_strategy: Optional[str] = None,
    winning_mae: Optional[float] = None,
    winning_mape: Optional[float] = None,
) -> float:
    """Compute a confidence score from the validation signal of whichever strategy actually won."""
    if pd.isna(model_mae):
        return 0.0

    if winning_strategy is None:
        winning_strategy = "lightgbm"

    if winning_strategy == "lightgbm":
        if pd.isna(baseline_mae):
            return 1.0

        best_simple_mae = float(baseline_mae)
        if pd.notna(trend_mae) and (pd.isna(best_simple_mae) or trend_mae < best_simple_mae):
            best_simple_mae = float(trend_mae)
        if pd.notna(rolling_mae) and (pd.isna(best_simple_mae) or rolling_mae < best_simple_mae):
            best_simple_mae = float(rolling_mae)

        improvement = best_simple_mae - float(model_mae)
        if improvement <= 0:
            base_confidence = 0.0
        else:
            denominator = max(abs(best_simple_mae), 1e-6)
            relative_improvement = improvement / denominator
            base_confidence = float(np.clip(relative_improvement / 0.2, 0.0, 1.0))
    else:
        capped_mape = winning_mape if pd.notna(winning_mape) else float("nan")
        if pd.isna(capped_mape):
            capped_mape = 100.0
        base_confidence = float(np.clip(1.0 - (float(capped_mape) / 200.0), 0.0, 1.0))

    if health is not None:
        health_score = float(health.get("health_score", 0.0) or 0.0)
        health_status = str(health.get("health_status", "") or "")
        health_reason = str(health.get("health_reason", "") or "")
        std = health.get("std")
        std_value = float(std) if std is not None and pd.notna(std) else float("nan")
        coefficient_of_variation = health.get("coefficient_of_variation")
        cov_value = float(coefficient_of_variation) if coefficient_of_variation is not None and pd.notna(coefficient_of_variation) else float("nan")
        is_degenerate_signal = False
        if health_status == "near_constant" or health_status == "too_short" or health_status == "empty":
            is_degenerate_signal = True
        elif pd.notna(std_value) and std_value <= 1e-8:
            is_degenerate_signal = True
        elif pd.notna(cov_value) and abs(cov_value) <= 1e-8:
            is_degenerate_signal = True
        elif not bool(health.get("is_healthy", True)) and health_score < 0.1 and "constant" in health_reason.lower():
            is_degenerate_signal = True

        if is_degenerate_signal:
            base_confidence = 0.1
        elif pd.notna(health_score):
            base_confidence = float(np.clip(0.7 * base_confidence + 0.3 * health_score, 0.0, 1.0))

    return base_confidence


def select_forecast_strategy_details(
    model_mae: float,
    baseline_mae: float,
    trend_mae: Optional[float] = None,
    rolling_mae: Optional[float] = None,
    min_improvement: float = 0.05,
    series_volatility: Optional[float] = None,
    series_scale: Optional[float] = None,
    series_length: Optional[int] = None,
    backtest_confidence: Optional[float] = None,
    health: Optional[Dict[str, object]] = None,
    baseline_mape: Optional[float] = None,
    trend_mape: Optional[float] = None,
    rolling_mape: Optional[float] = None,
) -> Dict[str, object]:
    """Choose the most reliable forecast strategy and explain why it was selected."""
    if pd.isna(model_mae):
        candidates = []
        if pd.notna(baseline_mae):
            candidates.append((baseline_mae, "baseline"))
        if pd.notna(trend_mae):
            candidates.append((trend_mae, "trend_baseline"))
        if pd.notna(rolling_mae):
            candidates.append((rolling_mae, "rolling_baseline"))
        if not candidates:
            computed_confidence = _compute_backtest_confidence(model_mae, baseline_mae, trend_mae=trend_mae, rolling_mae=rolling_mae, health=health)
            return {
                "strategy": "lightgbm",
                "selection_reason": "no baseline metrics were available",
                "relative_improvement": float("nan"),
                "best_simple_mae": float("nan"),
                "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "health_score": None,
            }
        best_mae, best_label = min(candidates, key=lambda item: item[0])
        strategy = best_label if best_mae < float("inf") else "baseline"
        computed_confidence = _compute_backtest_confidence(model_mae, baseline_mae, trend_mae=trend_mae, rolling_mae=rolling_mae, health=health)
        return {
            "strategy": strategy,
            "selection_reason": "model validation metrics were missing so the best simple baseline was selected",
            "relative_improvement": float("nan"),
            "best_simple_mae": float(best_mae),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
        }

    best_simple_mae = baseline_mae
    best_label = "baseline"
    if pd.notna(trend_mae) and (pd.isna(best_simple_mae) or trend_mae < best_simple_mae):
        best_simple_mae = trend_mae
        best_label = "trend_baseline"
    if pd.notna(rolling_mae) and (pd.isna(best_simple_mae) or rolling_mae < best_simple_mae):
        best_simple_mae = rolling_mae
        best_label = "rolling_baseline"

    selected_winning_mape = (
        baseline_mape if best_label == "baseline" else trend_mape if best_label == "trend_baseline" else rolling_mape
    )

    if pd.isna(best_simple_mae):
        computed_confidence = _compute_backtest_confidence(model_mae, baseline_mae, trend_mae=trend_mae, rolling_mae=rolling_mae, health=health, winning_strategy="lightgbm", winning_mae=best_simple_mae)
        return {
            "strategy": "lightgbm",
            "selection_reason": "no simple baseline could be evaluated",
            "relative_improvement": float("nan"),
            "best_simple_mae": float("nan"),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
        }

    if health is not None and not bool(health.get("is_healthy", True)):
        computed_confidence = _compute_backtest_confidence(
            model_mae,
            baseline_mae,
            trend_mae=trend_mae,
            rolling_mae=rolling_mae,
            health=health,
            winning_strategy=best_label,
            winning_mae=best_simple_mae,
            winning_mape=selected_winning_mape,
        )
        return {
            "strategy": best_label,
            "selection_reason": f"health gate rejected the model: {health.get('health_reason', 'weak signal')}",
            "relative_improvement": float("nan"),
            "best_simple_mae": float(best_simple_mae),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))),
        }

    if series_length is not None and int(series_length) < 40:
        computed_confidence = _compute_backtest_confidence(
            model_mae,
            baseline_mae,
            trend_mae=trend_mae,
            rolling_mae=rolling_mae,
            health=health,
            winning_strategy=best_label,
            winning_mae=best_simple_mae,
            winning_mape=selected_winning_mape,
        )
        return {
            "strategy": best_label,
            "selection_reason": "series was too short for a trained model to be trusted",
            "relative_improvement": float("nan"),
            "best_simple_mae": float(best_simple_mae),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
        }

    if series_volatility is not None:
        volatility = float(series_volatility)
        if pd.notna(volatility) and volatility > 1.0:
            computed_confidence = _compute_backtest_confidence(
                model_mae,
                baseline_mae,
                trend_mae=trend_mae,
                rolling_mae=rolling_mae,
                health=health,
                winning_strategy=best_label,
                winning_mae=best_simple_mae,
                winning_mape=selected_winning_mape,
            )
            return {
                "strategy": best_label,
                "selection_reason": "series volatility was too high for the trained model",
                "relative_improvement": float("nan"),
                "best_simple_mae": float(best_simple_mae),
                "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
            }

    if series_scale is not None:
        scale = float(series_scale)
        computed_confidence = _compute_backtest_confidence(
            model_mae,
            baseline_mae,
            trend_mae=trend_mae,
            rolling_mae=rolling_mae,
            health=health,
            winning_strategy=best_label,
            winning_mae=best_simple_mae,
            winning_mape=selected_winning_mape,
        )
        if pd.notna(scale) and abs(scale) > 1000.0:
            return {
                "strategy": best_label,
                "selection_reason": "series magnitude was too large for the trained model",
                "relative_improvement": float("nan"),
                "best_simple_mae": float(best_simple_mae),
                "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
            }
        return {
            "strategy": best_label,
            "selection_reason": "simple baseline outperformed the trained model",
            "relative_improvement": 0.0,
            "best_simple_mae": float(best_simple_mae),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
        }

    improvement = best_simple_mae - model_mae
    denominator = max(abs(best_simple_mae), 1e-6)
    relative_improvement = improvement / denominator
    if relative_improvement < min_improvement:
        computed_confidence = _compute_backtest_confidence(
            model_mae,
            baseline_mae,
            trend_mae=trend_mae,
            rolling_mae=rolling_mae,
            health=health,
            winning_strategy=best_label,
            winning_mae=best_simple_mae,
            winning_mape=selected_winning_mape,
        )
        return {
            "strategy": best_label,
            "selection_reason": "relative improvement was below the configured threshold",
            "relative_improvement": float(relative_improvement),
            "best_simple_mae": float(best_simple_mae),
            "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
        }

    computed_confidence = _compute_backtest_confidence(model_mae, baseline_mae, trend_mae=trend_mae, rolling_mae=rolling_mae, health=health, winning_strategy="lightgbm", winning_mae=best_simple_mae)
    if backtest_confidence is not None:
        confidence_value = float(backtest_confidence)
        if pd.notna(confidence_value) and confidence_value < DEFAULT_CONFIDENCE_THRESHOLD:
            return {
                "strategy": best_label,
                "selection_reason": "backtest confidence was too low for the trained model",
                "relative_improvement": float(relative_improvement),
                "best_simple_mae": float(best_simple_mae),
                "backtest_confidence": confidence_value,
                "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
            }

    return {
        "strategy": "lightgbm",
        "selection_reason": "model beat the best simple baseline by a meaningful margin and passed the health gate",
        "relative_improvement": float(relative_improvement),
        "best_simple_mae": float(best_simple_mae),
        "backtest_confidence": float(backtest_confidence) if backtest_confidence is not None else computed_confidence,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "health_score": float(health.get("health_score", float("nan"))) if health else float("nan"),
    }


def select_forecast_strategy(
    model_mae: float,
    baseline_mae: float,
    trend_mae: Optional[float] = None,
    rolling_mae: Optional[float] = None,
    min_improvement: float = 0.05,
    series_volatility: Optional[float] = None,
    series_scale: Optional[float] = None,
    series_length: Optional[int] = None,
    backtest_confidence: Optional[float] = None,
    health: Optional[Dict[str, object]] = None,
) -> str:
    """Choose the most reliable forecast strategy for a series based on validation error."""
    return str(
        select_forecast_strategy_details(
            model_mae,
            baseline_mae,
            trend_mae=trend_mae,
            rolling_mae=rolling_mae,
            min_improvement=min_improvement,
            series_volatility=series_volatility,
            series_scale=series_scale,
            series_length=series_length,
            backtest_confidence=backtest_confidence,
            health=health,
        )["strategy"]
    )


def _remove_path_safe(path: Path, retries: int = 5, delay: float = 0.2) -> None:
    def _onerror(func, path_str, exc_info):
        try:
            os.chmod(path_str, 0o700)
            func(path_str)
        except Exception:
            pass

    if not path.exists():
        return

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=_onerror)
            if not path.exists():
                return
        except Exception:
            pass
        time.sleep(delay)

    if path.exists():
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), 0o700)
                    os.remove(os.path.join(root, name))
                except Exception:
                    pass
            for name in dirs:
                try:
                    os.chmod(os.path.join(root, name), 0o700)
                    os.rmdir(os.path.join(root, name))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass


def split_by_host(input_csv: Path, output_dir: Path) -> List[Dict[str, object]]:
    """Split the large CSV into per-host files using chunked reads."""
    temp_dir = output_dir.parent / f".{output_dir.name}_tmp_{uuid.uuid4().hex}"
    _remove_path_safe(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    host_counts: Dict[str, int] = {}
    host_files: List[Dict[str, object]] = []

    reader = pd.read_csv(input_csv, chunksize=CHUNK_SIZE, low_memory=False)
    for chunk_idx, chunk in enumerate(reader, start=1):
        # Do NOT normalize here. Determine host column from raw chunk and
        # write raw rows to per-host temp files so normalization can run on
        # the full per-host series later.
        if chunk.empty:
            continue

        # Find host column candidates in the raw chunk
        host_col = _find_column([str(c) for c in chunk.columns], ["host_name", "host", "hostname", "server", "device"])
        if host_col is None:
            # Write entire chunk to a single 'unknown' host file
            host_name = "unknown"
            host_file = temp_dir / f"{sanitize_name(host_name)}.csv"
            host_file.parent.mkdir(parents=True, exist_ok=True)
            chunk.to_csv(host_file, mode="a", index=False, header=not host_file.exists())
            host_counts[host_name] = host_counts.get(host_name, 0) + len(chunk)
            if host_file not in [Path(item.get("file_path")) for item in host_files if "file_path" in item]:
                host_files.append({"host_name": host_name, "file_path": str(host_file), "row_count": len(chunk)})
            else:
                for item in host_files:
                    if item.get("host_name") == host_name:
                        item["row_count"] = int(item.get("row_count", 0)) + len(chunk)
                        break
            continue

        # Group by the raw host column and append raw rows to per-host files.
        for raw_host, host_chunk in chunk.groupby(host_col, sort=False):
            host_name = str(raw_host) if pd.notna(raw_host) else "unknown"
            host_file = temp_dir / f"{sanitize_name(host_name)}.csv"
            host_file.parent.mkdir(parents=True, exist_ok=True)

            host_chunk.to_csv(host_file, mode="a", index=False, header=not host_file.exists())
            host_counts[host_name] = host_counts.get(host_name, 0) + len(host_chunk)

            if host_file not in [Path(item.get("file_path")) for item in host_files if "file_path" in item]:
                host_files.append({"host_name": host_name, "file_path": str(host_file), "row_count": len(host_chunk)})
            else:
                for item in host_files:
                    if item.get("host_name") == host_name:
                        item["row_count"] = int(item.get("row_count", 0)) + len(host_chunk)
                        break

    print("\nPhase 1 summary:")
    print(f"{'host_name':<30} {'sanitized_file':<40} {'row_count'}")
    print("-" * 90)
    for host_name, row_count in sorted(host_counts.items()):
        file_name = f"{sanitize_name(host_name)}.csv"
        print(f"{host_name:<30} {file_name:<40} {row_count}")

    # Ensure any existing target host folder is removed before moving the temp directory.
    if output_dir.exists():
        _remove_path_safe(output_dir)

    try:
        shutil.move(str(temp_dir), str(output_dir))
    except Exception:
        # In case of intermittent Windows locking, try one more safe remove and move.
        if output_dir.exists():
            _remove_path_safe(output_dir)
        shutil.move(str(temp_dir), str(output_dir))

    # Update host file paths to their final locations after the move.
    for host_file in host_files:
        host_path = Path(host_file["file_path"])
        host_file["file_path"] = str(output_dir / host_path.name)

    # SECOND PASS: normalize each per-host CSV in full so per-(host,item)
    # timestamp filling sees the complete series rather than chunk-sized fragments.
    for host_entry in host_files:
        host_csv = Path(host_entry.get("file_path"))
        if not host_csv.exists():
            continue
        try:
            raw_df = pd.read_csv(host_csv, low_memory=False)
            normalized_df = normalize_metric_frame(raw_df, host_fallback=str(host_entry.get("host_name", "unknown")), item_fallback="unknown")
            normalized_df.to_csv(host_csv, index=False)
            # Update row_count to reflect post-normalization length
            host_entry["row_count"] = len(normalized_df)
            host_counts[host_entry.get("host_name")] = int(host_entry["row_count"])
        except Exception:
            # If normalization fails for a host, leave the raw file as-is and log.
            logging.exception(f"Failed to normalize host file: {host_csv}")

    return host_files


def count_training_tasks(host_files: List[Dict[str, object]]) -> int:
    total = 0
    for host_entry in host_files:
        host_csv = Path(host_entry["file_path"])
        if not host_csv.exists():
            continue
        try:
            host_name = str(host_entry.get("host_name", "unknown"))
            df = normalize_metric_frame(pd.read_csv(host_csv, low_memory=False), host_fallback=host_name, item_fallback="unknown")
            total += len(df.groupby("item_key", sort=False))
        except Exception:
            continue
    return total


def save_actual_vs_predicted_plot(host_name: str, item_key: str, timestamps, actuals, predictions, output_dir: Path) -> None:
    """Save a plot of actual versus predicted values over time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(timestamps, actuals, label="Actual", marker="o", linewidth=1.2)
    ax.plot(timestamps, predictions, label="Predicted", marker="x", linewidth=1.2)
    ax.set_title(f"{host_name} - {item_key}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def save_feature_importance_plot(model, feature_names: List[str], host_name: str, item_key: str, output_dir: Path) -> None:
    """Save a horizontal bar chart of feature importances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) == 0:
        return

    ranked_idx = np.argsort(importances)[::-1][:15]
    ranked_names = [feature_names[i] for i in ranked_idx]
    ranked_values = importances[ranked_idx]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.barh(ranked_names[::-1], ranked_values[::-1])
    ax.set_title(f"{host_name} - {item_key} feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)




def _write_audit_log(audit_log_path: Path, entry: Dict[str, object]) -> None:
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_entries: List[Dict[str, object]] = []
    if audit_log_path.exists():
        try:
            with audit_log_path.open("r", encoding="utf-8") as handle:
                existing_entries = json.load(handle)
        except Exception:
            existing_entries = []
    if not isinstance(existing_entries, list):
        existing_entries = []
    existing_entries.append(entry)
    with audit_log_path.open("w", encoding="utf-8") as handle:
        json.dump(existing_entries, handle, indent=2)


def _summarize_selection_reasons(row_records: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Count selection-reason outcomes from training decisions for easier monitoring."""
    summary: Dict[str, int] = {}
    for record in row_records:
        reason = str(record.get("selection_reason", "")).strip() or "unknown"
        summary[reason] = summary.get(reason, 0) + 1
    return summary


def _write_selection_reason_summary(reports_dir: Path, summary_df: pd.DataFrame, skipped_df: pd.DataFrame) -> None:
    """Write a compact CSV summary of selection and skip reasons for Power BI and monitoring."""
    rows: List[Dict[str, object]] = []
    if not summary_df.empty:
        fallback_rows = summary_df[summary_df.get("forecast_strategy", pd.Series(["lightgbm"] * len(summary_df))) != "lightgbm"]
        if not fallback_rows.empty:
            for reason, count in sorted(_summarize_selection_reasons([{"selection_reason": reason} for reason in fallback_rows["selection_reason"].fillna("unknown")]).items(), key=lambda item: (-item[1], item[0])):
                rows.append({"reason_type": "fallback_selection_reason", "reason": reason, "count": count})
    if not skipped_df.empty:
        for reason, count in sorted(_summarize_selection_reasons([{"selection_reason": reason} for reason in skipped_df.get("reason_skipped", pd.Series(["unknown"] * len(skipped_df))).fillna("unknown")]).items(), key=lambda item: (-item[1], item[0])):
            rows.append({"reason_type": "skipped_reason", "reason": reason, "count": count})
    if rows:
        pd.DataFrame(rows).to_csv(reports_dir / "selection_reason_summary.csv", index=False)


def train_models(
    host_files: List[Dict[str, object]],
    models_dir: Path,
    reports_dir: Path = REPORTS_DIR,
    plots_dir: Path = PLOTS_DIR,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    min_improvement: float = 0.05,
) -> List[Dict[str, object]]:
    """Train a LightGBM regressor for each numeric metric on each host."""
    if models_dir.exists():
        try:
            shutil.rmtree(models_dir)
        except Exception:
            for path in sorted(models_dir.glob("**/*"), reverse=True):
                try:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                except Exception:
                    pass
    models_dir.mkdir(parents=True, exist_ok=True)

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    skipped_series: List[Dict[str, object]] = []
    audit_log_path = reports_dir / "forecast_audit_log.json"
    fallback_report_rows: List[Dict[str, object]] = []

    for host_entry in host_files:
        host_name = str(host_entry["host_name"])
        host_csv = Path(host_entry["file_path"])

        if not host_csv.exists():
            continue

        df = normalize_metric_frame(pd.read_csv(host_csv, low_memory=False), host_fallback=host_name, item_fallback="unknown")
        df = df[df["value_type"].isin([0, 3])].copy()

        if df.empty:
            continue

        host_model_dir = models_dir / sanitize_name(host_name)
        host_model_dir.mkdir(parents=True, exist_ok=True)

        for item_key, group in df.groupby("item_key", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) > RECENT_WINDOW_SIZE:
                group = group.tail(RECENT_WINDOW_SIZE).copy()
            item_name = str(group["item_name"].iloc[0]) if not group["item_name"].empty else ""

            if len(group) < 30:
                reason = f"insufficient rows: {len(group)}"
                skipped_series.append(
                    {
                        "host_name": host_name,
                        "item_key": item_key,
                        "item_name": item_name,
                        "reason_skipped": reason,
                        "row_count": len(group),
                    }
                )
                print(f"Warning: skipping {host_name} / {item_key} with only {len(group)} rows")
                continue

            # Engineer lag, rolling-window, and temporal features from past values only.
            group = engineer_feature_frame(group.copy())

            # Create 7-day ahead targets (predict next 7 data points).
            feature_cols = build_feature_columns()
            
            target_cols = []
            for day_ahead in range(1, 8):
                target_col = f"target_{day_ahead}"
                group[target_col] = group["value"].shift(-day_ahead)
                target_cols.append(target_col)
            
            model_frame = group[["timestamp", *feature_cols, *target_cols]].dropna().reset_index(drop=True)

            if len(model_frame) < 10:
                reason = f"insufficient samples after feature engineering: {len(model_frame)}"
                skipped_series.append(
                    {
                        "host_name": host_name,
                        "item_key": item_key,
                        "item_name": item_name,
                        "reason_skipped": reason,
                        "row_count": len(group),
                    }
                )
                print(f"Warning: skipping {host_name} / {item_key} after feature engineering with only {len(model_frame)} samples")
                continue

            X = model_frame[feature_cols]
            y = model_frame[target_cols]  # Multi-output targets (7 columns)
            timestamps = model_frame["timestamp"]

            # Time-based split without shuffling.
            X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
                X,
                y,
                timestamps,
                test_size=0.2,
                shuffle=False,
            )

            # Train separate models for each day ahead for best accuracy
            models = []
            predictions_all = np.zeros((len(X_test), 7))
            
            for day_idx, target_col in enumerate(target_cols):
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(X_train, y_train.iloc[:, day_idx])
                predictions_all[:, day_idx] = model.predict(X_test)
                models.append(model)

            baseline_predictions = np.repeat(X_test["lag_1"].to_numpy(dtype=float)[:, None], 7, axis=1)
            trend_predictions = np.zeros((len(X_test), 7), dtype=float)
            rolling_predictions = np.zeros((len(X_test), 7), dtype=float)
            # Compute series statistics from the available (non-NaN) values.
            value_series = group["value"].dropna()
            series_length = int(len(value_series))
            series_mean = float(value_series.mean()) if len(value_series) else float("nan")
            series_std = float(value_series.std(ddof=0)) if len(value_series) else float("nan")
            series_volatility = float(series_std / max(abs(series_mean), 1e-6)) if pd.notna(series_mean) and pd.notna(series_std) and abs(series_mean) > 1e-6 else float("nan")
            series_scale = float(np.max(np.abs(value_series))) if len(value_series) else float("nan")
            health = evaluate_series_health(value_series)

            # Align baseline histories to the actual test timestamps. For each test
            # row we use all `group['value']` entries with timestamp strictly
            # before that test timestamp to simulate the real historical data
            # available at prediction time.
            ts_test_list = list(ts_test.reset_index(drop=True))
            for row_idx in range(len(ts_test_list)):
                test_ts = pd.Timestamp(ts_test_list[row_idx])
                hist_vals = group.loc[group["timestamp"] < test_ts, "value"].dropna().to_numpy(dtype=float)
                # If no history exists strictly before the test timestamp, fall
                # back to the available non-NaN values (or a zero) to keep the
                # baseline functions working (they already handle empty input).
                if hist_vals.size == 0:
                    hist_vals = value_series.to_numpy(dtype=float) if value_series.size > 0 else np.array([0.0], dtype=float)

                trend_predictions[row_idx, :] = np.asarray(generate_trend_forecast(hist_vals, days=7), dtype=float)
                rolling_predictions[row_idx, :] = np.asarray(generate_rolling_mean_forecast(hist_vals, days=7, window=3), dtype=float)

            model_metrics = [compute_error_metrics(y_test.iloc[:, i], predictions_all[:, i]) for i in range(7)]
            baseline_metrics = [compute_error_metrics(y_test.iloc[:, i], baseline_predictions[:, i]) for i in range(7)]
            trend_metrics = [compute_error_metrics(y_test.iloc[:, i], trend_predictions[:, i]) for i in range(7)]
            rolling_metrics = [compute_error_metrics(y_test.iloc[:, i], rolling_predictions[:, i]) for i in range(7)]
            
            mae = float(np.mean([metrics["mae"] for metrics in model_metrics]))
            rmse = float(np.mean([metrics["rmse"] for metrics in model_metrics]))
            mape = float(np.mean([metrics["mape"] for metrics in model_metrics if pd.notna(metrics["mape"])])) if any(pd.notna(metrics["mape"]) for metrics in model_metrics) else float("nan")
            metric_statuses = [str(metrics.get("mape_status", "ok")) for metrics in model_metrics]
            mape_status = "ok"
            if any(status == "undefined_zero_actuals" for status in metric_statuses):
                mape_status = "undefined_zero_actuals"
            mape_capped = float(np.clip(mape, 0.0, 200.0)) if pd.notna(mape) else float("nan")
            baseline_mae = float(np.mean([metrics["mae"] for metrics in baseline_metrics]))
            baseline_rmse = float(np.mean([metrics["rmse"] for metrics in baseline_metrics]))
            baseline_mape = float(np.mean([metrics["mape"] for metrics in baseline_metrics if pd.notna(metrics["mape"])])) if any(pd.notna(metrics["mape"]) for metrics in baseline_metrics) else float("nan")
            trend_mae = float(np.mean([metrics["mae"] for metrics in trend_metrics]))
            trend_rmse = float(np.mean([metrics["rmse"] for metrics in trend_metrics]))
            trend_mape = float(np.mean([metrics["mape"] for metrics in trend_metrics if pd.notna(metrics["mape"])])) if any(pd.notna(metrics["mape"]) for metrics in trend_metrics) else float("nan")
            rolling_mae = float(np.mean([metrics["mae"] for metrics in rolling_metrics]))
            rolling_rmse = float(np.mean([metrics["rmse"] for metrics in rolling_metrics]))
            rolling_mape = float(np.mean([metrics["mape"] for metrics in rolling_metrics if pd.notna(metrics["mape"])])) if any(pd.notna(metrics["mape"]) for metrics in rolling_metrics) else float("nan")
            strategy_details = select_forecast_strategy_details(
                mae,
                baseline_mae,
                trend_mae=trend_mae,
                rolling_mae=rolling_mae,
                min_improvement=min_improvement,
                series_volatility=series_volatility,
                series_scale=series_scale,
                series_length=series_length,
                backtest_confidence=None,
                health=health,
                baseline_mape=baseline_mape,
                trend_mape=trend_mape,
                rolling_mape=rolling_mape,
            )
            forecast_strategy = str(strategy_details["strategy"])
            selection_reason = str(strategy_details.get("selection_reason", ""))
            if forecast_strategy != "lightgbm":
                fallback_report_rows.append(
                    {
                        "host_name": host_name,
                        "item_key": item_key,
                        "item_name": item_name,
                        "strategy": forecast_strategy,
                        "reason": selection_reason,
                        "health_score": strategy_details.get("health_score"),
                        "backtest_confidence": strategy_details.get("backtest_confidence"),
                        "series_length": series_length,
                        "series_scale": series_scale,
                        "series_volatility": series_volatility,
                    }
                )

            # Save all 7 models
            model_paths = []
            for day_idx, model in enumerate(models):
                model_path = host_model_dir / f"{sanitize_name(str(item_key))}_day{day_idx+1}.txt"
                model.booster_.save_model(str(model_path))
                model_paths.append(str(model_path))

            if progress_callback is not None:
                progress_callback(len(results) + len(skipped_series), f"Trained {item_key} for {host_name}")
            
            # Store primary model path (day 1)
            primary_model_path = model_paths[0]

            results.append(
                {
                    "host_name": host_name,
                    "item_key": item_key,
                    "item_name": item_name,
                    "num_samples": len(model_frame),
                    "num_train": len(X_train),
                    "num_test": len(X_test),
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "mape_capped": mape_capped,
                    "mape_status": mape_status,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "baseline_mape": baseline_mape,
                    "trend_mae": trend_mae,
                    "trend_rmse": trend_rmse,
                    "trend_mape": trend_mape,
                    "rolling_mae": rolling_mae,
                    "rolling_rmse": rolling_rmse,
                    "rolling_mape": rolling_mape,
                    "forecast_strategy": forecast_strategy,
                    "selection_reason": selection_reason,
                    "health_score": strategy_details.get("health_score"),
                    "backtest_confidence": strategy_details.get("backtest_confidence"),
                    "confidence_threshold": strategy_details.get("confidence_threshold"),
                    "model_path": Path(primary_model_path).name,
                }
            )

            _write_audit_log(
                audit_log_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "host_name": host_name,
                    "item_key": item_key,
                    "item_name": item_name,
                    "strategy": forecast_strategy,
                    "selection_reason": selection_reason,
                    "health_score": strategy_details.get("health_score"),
                    "backtest_confidence": strategy_details.get("backtest_confidence"),
                    "series_length": series_length,
                    "series_scale": series_scale,
                    "series_volatility": series_volatility,
                },
            )

            plot_dir = plots_dir / sanitize_name(host_name) / sanitize_name(str(item_key))
            try:
                validation_predictions = select_validation_predictions(
                    forecast_strategy,
                    model_predictions=predictions_all[:, 0],
                    baseline_predictions=baseline_predictions[:, 0],
                    trend_predictions=trend_predictions[:, 0],
                    rolling_predictions=rolling_predictions[:, 0],
                )
                save_validation_predictions_csv(
                    host_name,
                    item_key,
                    ts_test,
                    y_test.iloc[:, 0],
                    validation_predictions,
                    plot_dir,
                )
                save_actual_vs_predicted_plot(host_name, item_key, ts_test, y_test.iloc[:, 0], validation_predictions, plot_dir)
            except Exception as exc:
                print(f"Warning: failed to create actual/predicted plot for {host_name} / {item_key}: {exc}")

            try:
                # Use first model for feature importance
                save_feature_importance_plot(models[0], feature_cols, host_name, item_key, plot_dir)
            except Exception as exc:
                print(f"Warning: failed to create feature importance plot for {host_name} / {item_key}: {exc}")

    summary_df = pd.DataFrame(results)
    skipped_df = pd.DataFrame(skipped_series)

    if not summary_df.empty:
        selection_summary = _summarize_selection_reasons([{"selection_reason": reason} for reason in summary_df["selection_reason"].fillna("unknown")])
        print("\nSelection reason summary:")
        for reason, count in sorted(selection_summary.items(), key=lambda item: (-item[1], item[0])):
            print(f"  - {reason}: {count}")

    if not skipped_df.empty:
        skip_reason_summary = _summarize_selection_reasons([{"selection_reason": reason} for reason in skipped_df.get("reason_skipped", pd.Series(["unknown"] * len(skipped_df))).fillna("unknown")])
        print("\nSkipped reason summary:")
        for reason, count in sorted(skip_reason_summary.items(), key=lambda item: (-item[1], item[0])):
            print(f"  - {reason}: {count}")

    _write_selection_reason_summary(reports_dir, summary_df, skipped_df)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("rmse", ascending=True).reset_index(drop=True)
        summary_df.to_csv(reports_dir / "training_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["host_name", "item_key", "item_name", "num_samples", "num_train", "num_test", "mae", "rmse", "mape", "mape_capped", "mape_status", "baseline_mae", "baseline_rmse", "baseline_mape", "trend_mae", "trend_rmse", "trend_mape", "rolling_mae", "rolling_rmse", "rolling_mape", "forecast_strategy", "selection_reason", "health_score", "backtest_confidence", "confidence_threshold", "model_path"]).to_csv(reports_dir / "training_summary.csv", index=False)

    if not skipped_df.empty:
        skipped_df.to_csv(reports_dir / "skipped_series.csv", index=False)
    else:
        pd.DataFrame(columns=["host_name", "item_key", "item_name", "reason_skipped", "row_count"]).to_csv(reports_dir / "skipped_series.csv", index=False)

    validation_path = reports_dir / "training_summary.csv"
    validate_backtest_confidence(str(validation_path))

    fallback_df = pd.DataFrame(fallback_report_rows)
    if not fallback_df.empty:
        fallback_df.to_csv(reports_dir / "fallback_report.csv", index=False)
    else:
        pd.DataFrame(columns=["host_name", "item_key", "item_name", "strategy", "reason", "health_score", "backtest_confidence", "series_length", "series_scale", "series_volatility"]).to_csv(reports_dir / "fallback_report.csv", index=False)

    print("\nPhase 2 summary:")
    print(f"{'host_name':<30} {'item_key':<50} {'num_samples':>12} {'mae':>12} {'rmse':>12}")
    print("-" * 130)
    for row in results:
        print(
            f"{row['host_name']:<30} {str(row['item_key']):<50} {row['num_samples']:>12} {row['mae']:>12.4f} {row['rmse']:>12.4f}"
        )

    print("\nFinal reporting summary:")
    print(f"Total models trained: {len(results)}")
    print(f"Total series skipped: {len(skipped_series)}")
    if not summary_df.empty:
        print(f"Average MAE: {summary_df['mae'].mean():.4f}")
        print(f"Average RMSE: {summary_df['rmse'].mean():.4f}")
        print("Top 3 best by RMSE:")
        for _, row in summary_df.head(3).iterrows():
            print(f"  - {row['host_name']} / {row['item_key']}: RMSE={row['rmse']:.4f}")
        print("Top 3 worst by RMSE:")
        for _, row in summary_df.sort_values("rmse", ascending=False).head(3).iterrows():
            print(f"  - {row['host_name']} / {row['item_key']}: RMSE={row['rmse']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models and validate backtest confidence")
    parser.add_argument("--validate", action="store_true", help="validate backtest_confidence from an existing training_summary.csv")
    parser.add_argument("--training-summary-path", default=str(REPORTS_DIR / "training_summary.csv"), help="Path to the training summary CSV to validate")
    args = parser.parse_args()

    if args.validate:
        validate_backtest_confidence(args.training_summary_path)
    else:
        host_files = split_by_host(INPUT_CSV, HOSTS_DIR)
        train_models(host_files, MODELS_DIR)
