"""
State Normalization for Trading Environments
Extracted from utils.py for cleaner imports
"""
import numpy as np
import pandas as pd
from config.settings import TrainingConfig


# Per-timestep feature columns (order matters) used by training + live inference.
# These correspond to the 11 features stacked in `normalize_state()`.
WINDOW_FEATURE_COLUMNS = (
    'Close',
    'rsi',
    'stoch_k',
    'macd_diff',
    'bb_pband',
    'atr',
    'volume_ratio',
    'adx',
    'price_vs_sma20',
    'price_vs_sma50',
    'sma20_slope',
)


def normalize_state(state_df, current_step, window_size=None):
    """
    Extracts and normalizes the window of data for the agent.
    Returns a comprehensive feature vector.
    """
    if window_size is None:
        window_size = TrainingConfig.WINDOW_SIZE
        
    # Get the window of data
    start_idx = max(0, current_step - window_size + 1)
    end_idx = current_step + 1
    window = state_df.iloc[start_idx:end_idx].copy()
    
    # Pad if window is too small
    if len(window) < window_size:
        padding = pd.DataFrame(
            np.zeros((window_size - len(window), len(window.columns))),
            columns=window.columns
        )
        window = pd.concat([padding, window], ignore_index=True)
    
    # Get baseline price for normalization
    baseline_price = window['Close'].iloc[0]
    if baseline_price == 0:
        baseline_price = 1.0
    
    # FEATURE SET 1: Price Action
    norm_close = (window['Close'].values / baseline_price) - 1.0
    
    # FEATURE SET 2: Momentum Indicators
    norm_rsi = window['rsi'].values / 100.0
    norm_stoch_k = window['stoch_k'].values / 100.0
    
    # FEATURE SET 3: Trend Indicators
    norm_macd_diff = window['macd_diff'].values / window['Close'].values
    norm_adx = window['adx'].values / 100.0
    norm_price_sma20 = np.clip(window['price_vs_sma20'].values, -0.5, 0.5)
    norm_price_sma50 = np.clip(window['price_vs_sma50'].values, -0.5, 0.5)
    norm_sma20_slope = np.clip(window['sma20_slope'].values * 10, -1, 1)
    
    # FEATURE SET 4: Volatility
    norm_bb_pband = np.clip(window['bb_pband'].values, -0.5, 1.5)
    norm_atr = window['atr'].values / window['Close'].values
    
    # FEATURE SET 5: Volume
    norm_volume_ratio = np.clip(window['volume_ratio'].values, 0, 5) / 5.0
    
    # FEATURE SET 6: Market Regime (Latest only)
    latest = window.iloc[-1]
    regime_features = np.array([
        latest['regime'],
        latest['trend_strength'],
        latest['sma_cross'],
        np.clip(latest['momentum_5d'] * 10, -1, 1),
        np.clip(latest['momentum_10d'] * 5, -1, 1),
        np.clip(latest['momentum_20d'] * 3, -1, 1),
    ])
    
    # Combine all window features
    window_features = np.column_stack((
        norm_close,
        norm_rsi,
        norm_stoch_k,
        norm_macd_diff,
        norm_bb_pband,
        norm_atr,
        norm_volume_ratio,
        norm_adx,
        norm_price_sma20,
        norm_price_sma50,
        norm_sma20_slope,
    ))
    
    # Flatten and add regime features
    features = np.concatenate([
        window_features.flatten(),
        regime_features
    ])
    
    return features.astype(np.float32)


def normalize_window_state(state_df, current_step, window_size=None):
    """Normalize and return window-only features (no regime/portfolio).

    This matches the V2 Transformer Dueling DQN training/inference contract used by
    GPU training (`scripts/train_gpu.py`) and the live bots: a flattened
    `window_size * NUM_WINDOW_FEATURES` vector.
    """
    if window_size is None:
        window_size = TrainingConfig.WINDOW_SIZE

    start_idx = max(0, current_step - window_size + 1)
    end_idx = current_step + 1
    window = state_df.iloc[start_idx:end_idx].copy()

    if len(window) < window_size:
        padding = pd.DataFrame(
            np.zeros((window_size - len(window), len(window.columns))),
            columns=window.columns,
        )
        window = pd.concat([padding, window], ignore_index=True)

    baseline_price = window['Close'].iloc[0]
    if baseline_price == 0:
        baseline_price = 1.0

    norm_close = (window['Close'].values / baseline_price) - 1.0
    norm_rsi = window['rsi'].values / 100.0
    norm_stoch_k = window['stoch_k'].values / 100.0
    norm_macd_diff = window['macd_diff'].values / window['Close'].values
    norm_bb_pband = np.clip(window['bb_pband'].values, -0.5, 1.5)
    norm_atr = window['atr'].values / window['Close'].values
    norm_volume_ratio = np.clip(window['volume_ratio'].values, 0, 5) / 5.0
    norm_adx = window['adx'].values / 100.0
    norm_price_sma20 = np.clip(window['price_vs_sma20'].values, -0.5, 0.5)
    norm_price_sma50 = np.clip(window['price_vs_sma50'].values, -0.5, 0.5)
    norm_sma20_slope = np.clip(window['sma20_slope'].values * 10, -1, 1)

    window_features = np.column_stack((
        norm_close,
        norm_rsi,
        norm_stoch_k,
        norm_macd_diff,
        norm_bb_pband,
        norm_atr,
        norm_volume_ratio,
        norm_adx,
        norm_price_sma20,
        norm_price_sma50,
        norm_sma20_slope,
    ))

    return window_features.flatten().astype(np.float32)


def get_state_size(window_size=None):
    """Calculate the total state size for the agent."""
    if window_size is None:
        window_size = TrainingConfig.WINDOW_SIZE
        
    num_window_features = TrainingConfig.NUM_WINDOW_FEATURES
    num_regime_features = TrainingConfig.NUM_REGIME_FEATURES
    num_portfolio_features = TrainingConfig.NUM_PORTFOLIO_FEATURES
    
    return (window_size * num_window_features) + num_regime_features + num_portfolio_features
