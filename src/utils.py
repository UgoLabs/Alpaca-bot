"""
Technical Indicators and State Normalization for Swing Trading
Rewritten to use pure Pandas/Numpy (Removing 'ta' library dependency)
"""

import pandas as pd
import numpy as np

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1)
    return 100 - (100 / (1 + rs))

def calculate_adx(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's Smoothing (alpha = 1/n)
    tr_s = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    
    # Avoid division by zero
    tr_s = tr_s.replace(0, 1)
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    denom = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / denom.replace(0, 1)
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    return adx

def calculate_macd(close, slow=26, fast=12, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    diff = macd - signal_line
    return macd, signal_line, diff

def calculate_bollinger_bands(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

def calculate_stoch(high, low, close, window=14, smooth_window=3):
    s_low = low.rolling(window=window).min()
    s_high = high.rolling(window=window).max()
    k = 100 * ((close - s_low) / (s_high - s_low).replace(0, 1))
    d = k.rolling(window=smooth_window).mean()
    return k, d

def add_technical_indicators(df):
    """
    Adds comprehensive technical indicators to the DataFrame.
    Expects columns: 'Open', 'High', 'Low', 'Close', 'Volume'
    """
    df = df.copy()
    
    # Handle MultiIndex and basic cleanup
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    df = df.loc[:, ~df.columns.duplicated()]
    if 'Close' in df.columns: df['Close'] = df['Close'].replace(0, np.nan).ffill()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].replace(0, np.nan).ffill()

    # Get Series
    close = df['Close'] if 'Close' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    # 1. TREND
    df['sma_10'] = close.rolling(window=10).mean().fillna(0)
    df['sma_20'] = close.rolling(window=20).mean().fillna(0)
    df['sma_50'] = close.rolling(window=50).mean().fillna(0)
    df['sma_200'] = close.rolling(window=200).mean().fillna(0)
    
    df['ema_12'] = close.ewm(span=12, adjust=False).mean().fillna(0)
    df['ema_26'] = close.ewm(span=26, adjust=False).mean().fillna(0)
    
    macd, macd_signal, macd_diff = calculate_macd(close)
    df['macd'] = macd.fillna(0)
    df['macd_signal'] = macd_signal.fillna(0)
    df['macd_diff'] = macd_diff.fillna(0)
    
    df['adx'] = calculate_adx(high, low, close).fillna(0)
    
    # 2. MOMENTUM
    df['rsi'] = calculate_rsi(close).fillna(50)
    
    stoch_k, stoch_d = calculate_stoch(high, low, close)
    df['stoch_k'] = stoch_k.fillna(50)
    df['stoch_d'] = stoch_d.fillna(50)
    
    # Williams %R
    highest_high = high.rolling(window=14).max()
    lowest_low = low.rolling(window=14).min()
    df['williams_r'] = (-100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)).fillna(-50)
    
    # ROC
    df['roc'] = close.pct_change(periods=10).fillna(0) * 100

    # 3. VOLATILITY
    bb_mid, bb_high, bb_low = calculate_bollinger_bands(close)
    df['bb_mid'] = bb_mid.fillna(0)
    df['bb_high'] = bb_high.fillna(0)
    df['bb_low'] = bb_low.fillna(0)
    df['bb_width'] = ((bb_high - bb_low) / bb_mid.replace(0, 1)).fillna(0) * 100
    df['bb_pband'] = ((close - bb_low) / (bb_high - bb_low).replace(0, 1)).fillna(0.5)

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0)

    # 4. VOLUME
    df['volume_sma'] = volume.rolling(window=20).mean().fillna(0)
    df['volume_ratio'] = (volume / df['volume_sma'].replace(0, 1)).fillna(1)
    
    # OBV
    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # MFI (Simplified)
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, 1)
    df['mfi'] = (100 - (100 / (1 + mfr))).fillna(50)

    # 5. DERIVED
    df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20'].replace(0, 1)
    df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50'].replace(0, 1)
    df['price_vs_sma200'] = (close - df['sma_200']) / df['sma_200'].replace(0, 1)
    
    df['sma20_slope'] = df['sma_20'].pct_change(periods=5).fillna(0)
    df['sma50_slope'] = df['sma_50'].pct_change(periods=10).fillna(0)
    
    df['sma_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    df['volatility_regime'] = df['atr'] / close.replace(0, 1)
    
    df['momentum_5d'] = close.pct_change(periods=5).fillna(0)
    df['momentum_10d'] = close.pct_change(periods=10).fillna(0)
    df['momentum_20d'] = close.pct_change(periods=20).fillna(0)
    df['return_volatility'] = close.pct_change().rolling(window=20).std().fillna(0)

    # 6. MARKET REGIME
    df['regime'] = np.where(
        (close > df['sma_50']) & (df['sma_50'] > df['sma_200']), 1,
        np.where((close < df['sma_50']) & (df['sma_50'] < df['sma_200']), -1, 0)
    )
    df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)

    return df

def normalize_state(state_df, current_step, window_size):
    """
    Extracts and normalizes the window of data for the agent.
    Returns a comprehensive feature vector.
    """
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
    
    # =========================================================================
    # FEATURE SET 1: Price Action (Window of data)
    # =========================================================================
    # Normalized price (relative to window start)
    norm_close = (window['Close'].values / baseline_price) - 1.0
    
    # =========================================================================
    # FEATURE SET 2: Momentum Indicators
    # =========================================================================
    norm_rsi = window['rsi'].values / 100.0  # 0-1 scale
    norm_stoch_k = window['stoch_k'].values / 100.0  # 0-1 scale
    norm_stoch_d = window['stoch_d'].values / 100.0  # 0-1 scale
    norm_williams = (window['williams_r'].values + 100) / 100.0  # -100 to 0 -> 0 to 1
    norm_mfi = window['mfi'].values / 100.0  # 0-1 scale
    
    # =========================================================================
    # FEATURE SET 3: Trend Indicators
    # =========================================================================
    # MACD relative to price
    norm_macd = window['macd'].values / window['Close'].values
    norm_macd_signal = window['macd_signal'].values / window['Close'].values
    norm_macd_diff = window['macd_diff'].values / window['Close'].values
    
    # ADX (already 0-100, normalize to 0-1)
    norm_adx = window['adx'].values / 100.0
    
    # Price vs SMAs (already percentage)
    norm_price_sma20 = np.clip(window['price_vs_sma20'].values, -0.5, 0.5)
    norm_price_sma50 = np.clip(window['price_vs_sma50'].values, -0.5, 0.5)
    
    # SMA slopes (clip extreme values)
    norm_sma20_slope = np.clip(window['sma20_slope'].values * 10, -1, 1)
    norm_sma50_slope = np.clip(window['sma50_slope'].values * 10, -1, 1)
    
    # =========================================================================
    # FEATURE SET 4: Volatility
    # =========================================================================
    # Bollinger %B (already normalized ~0-1, but can exceed)
    norm_bb_pband = np.clip(window['bb_pband'].values, -0.5, 1.5)
    
    # ATR relative to price
    norm_atr = window['atr'].values / window['Close'].values
    
    # Volatility regime
    norm_vol_regime = np.clip(window['volatility_regime'].values * 20, 0, 2)
    
    # =========================================================================
    # FEATURE SET 5: Volume
    # =========================================================================
    # Volume ratio (relative to 20-day average)
    norm_volume_ratio = np.clip(window['volume_ratio'].values, 0, 5) / 5.0
    
    # =========================================================================
    # FEATURE SET 6: Market Regime (Single Values - Latest)
    # =========================================================================
    latest = window.iloc[-1]
    regime_features = np.array([
        latest['regime'],                    # -1, 0, 1
        latest['trend_strength'],            # 0 or 1
        latest['sma_cross'],                 # -1 or 1
        np.clip(latest['momentum_5d'] * 10, -1, 1),   # 5-day momentum
        np.clip(latest['momentum_10d'] * 5, -1, 1),   # 10-day momentum
        np.clip(latest['momentum_20d'] * 3, -1, 1),   # 20-day momentum
    ])
    
    # =========================================================================
    # COMBINE ALL FEATURES
    # =========================================================================
    # Window features (each is window_size length)
    window_features = np.column_stack((
        norm_close,           # Price action
        norm_rsi,             # RSI
        norm_stoch_k,         # Stochastic %K
        norm_macd_diff,       # MACD histogram
        norm_bb_pband,        # Bollinger %B
        norm_atr,             # ATR ratio
        norm_volume_ratio,    # Volume ratio
        norm_adx,             # Trend strength
        norm_price_sma20,     # Price vs SMA20
        norm_price_sma50,     # Price vs SMA50
        norm_sma20_slope,     # SMA20 direction
    ))
    
    # Flatten window features and add regime features
    features = np.concatenate([
        window_features.flatten(),
        regime_features
    ])
    
    return features.astype(np.float32)


def get_state_size(window_size):
    """Calculate the total state size for the agent."""
    num_window_features = 11  # Features per timestep
    num_regime_features = 6   # Single-value regime features
    num_portfolio_features = 5  # Added in environment
    
    return (window_size * num_window_features) + num_regime_features + num_portfolio_features


def detect_market_regime(df, current_step):
    """
    Detect the current market regime with faster reaction and chop detection.
    
    Returns:
        regime: 'bullish', 'bearish', 'neutral', or 'choppy'
        strength: float 0-1 indicating regime strength
        is_choppy: bool indicating if market is choppy/sideways
    """
    if current_step < 50:
        return 'neutral', 0.5, False
    
    row = df.iloc[current_step]
    
    # Check if indicators exist
    if 'sma_50' not in df.columns:
        return 'neutral', 0.5, False
    
    close = row['Close']
    sma_50 = row['sma_50']
    sma_200 = row.get('sma_200', sma_50)
    adx = row.get('adx', 20)
    bb_width = row.get('bb_width', 1.0)
    
    # 1. Detect Choppy Market (Sideways Death Trap)
    # Low ADX (<20) usually means choppy/coiling
    is_choppy = adx < 20
    
    # 2. Faster Trend Detection
    bullish_score = 0
    
    # Primary Trend (Fast)
    if close > sma_50:
        bullish_score += 1
    
    # Secondary Trend (Slow/Background)
    if close > sma_200:
        bullish_score += 1
        
    # Moving Average Alignment (Golden Cross) - Bonus, not requirement
    if sma_50 > sma_200:
        bullish_score += 0.5
        
    # Momentum (RSI)
    rsi = row.get('rsi', 50)
    if rsi > 50:
        bullish_score += 0.5
    
    # Determine Regime
    if bullish_score >= 2.0:
        regime = 'bullish'
        strength = 1.0
    elif bullish_score >= 1.0:
        regime = 'neutral_bullish' # Weak bull / recovery
        strength = 0.6
    else:
        regime = 'bearish'
        strength = 0.2
        
    return regime, strength, is_choppy
