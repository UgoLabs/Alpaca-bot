"""
Canonical feature column lists for training, backtest, and live inference.
"""
from __future__ import annotations

# Same 11 features as swing training / SWING_MODEL_PATH checkpoints.
SWING_FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "rsi",
    "macd",
    "macd_signal",
    "adx",
    "sma_20",
    "ema_12",
]


def get_feature_cols(feature_set: str = "swing") -> list[str]:
    """feature_set: 'swing' (11 daily features)."""
    fs = (feature_set or "swing").strip().lower()
    if fs in ("swing", "11"):
        return list(SWING_FEATURE_COLS)
    raise ValueError(f"Unknown feature_set={feature_set!r}. Use 'swing'.")


def feature_set_dim(feature_set: str = "swing") -> int:
    return len(get_feature_cols(feature_set))
