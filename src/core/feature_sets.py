"""
Canonical feature column lists for training, backtest, and live inference.
Keep swing (11) and day (15) definitions in one place.
"""
from __future__ import annotations

from src.core.indicators import DAY_FEATURE_COLS  # noqa: F401 — re-export

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


def get_feature_cols(feature_set: str = "day") -> list[str]:
    """feature_set: 'day' (15 intraday) | 'swing' (11, for transfer from swing brain)."""
    fs = (feature_set or "day").strip().lower()
    if fs in ("swing", "11"):
        return list(SWING_FEATURE_COLS)
    if fs in ("day", "15", "intraday"):
        return list(DAY_FEATURE_COLS)
    raise ValueError(f"Unknown feature_set={feature_set!r}. Use 'day' or 'swing'.")


def feature_set_dim(feature_set: str = "day") -> int:
    return len(get_feature_cols(feature_set))
