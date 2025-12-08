"""
Day Trading Bot (Paper Account 2)
- Scans every 1 minute
- Uses 15-minute candles for analysis
- Closes all positions at EOD (3:55 PM ET)
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, time as dtime
import pytz
from swing_model import DuelingDQN

# =============================================================================
# CONFIGURATION
# =============================================================================
API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
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
    
    tr_s = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1/window, adjust=False).mean()
    
    tr_s = tr_s.replace(0, 1)
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    denom = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / denom.replace(0, 1)
    adx = dx.ewm(alpha=1/window, adjust=False).mean()
    return adx

def add_technical_indicators(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    df = df.loc[:, ~df.columns.duplicated()]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Trend
    df['sma_20'] = close.rolling(window=20).mean().fillna(0)
    df['sma_50'] = close.rolling(window=50).mean().fillna(0)
    df['sma_200'] = close.rolling(window=200).mean().fillna(0)
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = (exp1 - exp2).fillna(0)
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)
    df['macd_diff'] = (df['macd'] - df['macd_signal']).fillna(0)
    
    # ADX & RSI
    df['adx'] = calculate_adx(high, low, close).fillna(0)
    df['rsi'] = calculate_rsi(close).fillna(50)
    
    # Stochastic
    s_low = low.rolling(window=14).min()
    s_high = high.rolling(window=14).max()
    df['stoch_k'] = (100 * ((close - s_low) / (s_high - s_low).replace(0, 1))).fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean().fillna(50)
    
    # Bollinger Bands
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    df['bb_mid'] = sma.fillna(0)
    df['bb_high'] = (sma + 2*std).fillna(0)
    df['bb_low'] = (sma - 2*std).fillna(0)
    df['bb_width'] = ((df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, 1)).fillna(0) * 100
    df['bb_pband'] = ((close - df['bb_low']) / (df['bb_high'] - df['bb_low']).replace(0, 1)).fillna(0.5)

    # ATR & Volume
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0)
    df['volume_sma'] = volume.rolling(window=20).mean().fillna(0)
    df['volume_ratio'] = (volume / df['volume_sma'].replace(0, 1)).fillna(1)
    
    # Computed Features
    df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20'].replace(0, 1)
    df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50'].replace(0, 1)
    df['sma20_slope'] = df['sma_20'].pct_change(periods=5).fillna(0)
    df['sma50_slope'] = df['sma_50'].pct_change(periods=10).fillna(0)
    df['sma_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    df['momentum_5d'] = close.pct_change(periods=5).fillna(0)
    df['momentum_10d'] = close.pct_change(periods=10).fillna(0)
    df['momentum_20d'] = close.pct_change(periods=20).fillna(0)
    
    df['regime'] = np.where(
        (close > df['sma_50']) & (df['sma_50'] > df['sma_200']), 1,
        np.where((close < df['sma_50']) & (df['sma_50'] < df['sma_200']), -1, 0)
    )
    df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)
    
    return df

def normalize_state(state_df, current_step, window_size=20):
    start_idx = max(0, current_step - window_size + 1)
    end_idx = current_step + 1
    window = state_df.iloc[start_idx:end_idx].copy()
    
    if len(window) < window_size:
        padding = pd.DataFrame(np.zeros((window_size - len(window), len(window.columns))), columns=window.columns)
        window = pd.concat([padding, window], ignore_index=True)
    
    baseline_price = window['close'].iloc[0] if window['close'].iloc[0] != 0 else 1.0
    
    norm_close = (window['close'].values / baseline_price) - 1.0
    norm_rsi = window['rsi'].values / 100.0
    norm_stoch_k = window['stoch_k'].values / 100.0
    norm_macd_diff = window['macd_diff'].values / window['close'].values
    norm_bb_pband = np.clip(window['bb_pband'].values, -0.5, 1.5)
    norm_atr = window['atr'].values / window['close'].values
    norm_volume_ratio = np.clip(window['volume_ratio'].values, 0, 5) / 5.0
    norm_adx = window['adx'].values / 100.0
    norm_price_sma20 = np.clip(window['price_vs_sma20'].values, -0.5, 0.5)
    norm_price_sma50 = np.clip(window['price_vs_sma50'].values, -0.5, 0.5)
    norm_sma20_slope = np.clip((window['sma_20'].pct_change(periods=5).fillna(0).values) * 10, -1, 1)
    
    latest = window.iloc[-1]
    regime_features = np.array([
        latest['regime'],
        latest['trend_strength'],
        latest['sma_cross'],
        np.clip(latest['momentum_5d'] * 10, -1, 1),
        np.clip(latest['momentum_10d'] * 5, -1, 1),
        np.clip(latest['momentum_20d'] * 3, -1, 1),
    ])
    
    window_features = np.column_stack((
        norm_close, norm_rsi, norm_stoch_k, norm_macd_diff, norm_bb_pband,
        norm_atr, norm_volume_ratio, norm_adx, norm_price_sma20, norm_price_sma50, norm_sma20_slope
    ))
    
    features = np.concatenate([window_features.flatten(), regime_features])
    return features.astype(np.float32)

# =============================================================================
# DAY TRADER CLASS
# =============================================================================

class DayTrader:
    def __init__(self, model_path='models/live_model.pth'):
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        state_size = 231
        self.agent = DuelingDQN(state_size, 3, use_noisy=True)
        self.agent.load(model_path)
        self.agent.model.eval()
        
        self.symbols = self.load_day_trade_list()
        print(f"🚀 Day Trader initialized (Account: {BASE_URL})")
        print(f"📋 Watchlist: {len(self.symbols)} top movers")

    def load_day_trade_list(self):
        try:
            with open('../day_trade_list.txt', 'r') as f:
                return [l.strip() for l in f if l.strip()]
        except:
            return ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'SPY', 'QQQ']

    def check_eod_liquidation(self):
        # Close all positions at 3:55 PM ET
        now = datetime.now(pytz.timezone('US/Eastern'))
        if now.hour == 15 and now.minute >= 55:
            print("🕒 EOD Reached (3:55 PM ET). Liquidating all positions...")
            self.api.close_all_positions()
            return True
        return False

    def get_market_data(self, symbol):
        try:
            # Get 15Min bars for the last 5 days
            bars = self.api.get_bars(symbol, '15Min', limit=200, feed='iex').df
            if bars is None or len(bars) < 50: return None
            return bars
        except: return None

    def trade_loop(self):
        print(f"⏱️  Scanning every 60 seconds...")
        while True:
            # EOD Check
            if self.check_eod_liquidation():
                print("💤 Market closing soon. Sleeping until tomorrow.")
                time.sleep(3600*12) # Sleep 12 hours
                continue

            # Skip if market closed
            try:
                if not self.api.get_clock().is_open:
                    print("💤 Market closed. Waiting...")
                    time.sleep(300)
                    continue
            except: pass

            unique_decisions = []
            
            for symbol in self.symbols:
                try:
                    # 1. Get Historical Data (15Min Bars)
                    bars = self.get_market_data(symbol)
                    if bars is None: continue
                    
                    # 2. Get Real-Time Price (IEX)
                    info = {}
                    current_price = float(bars['close'].iloc[-1]) # Default to last bar close
                    try:
                        trade = self.api.get_latest_trade(symbol, feed='iex')
                        current_price = float(trade.price)
                        # info['iex_price'] = current_price
                        # info['iex_time'] = trade.timestamp
                    except: pass
                    
                    # Prepare Data (Indicators on 15Min bars)
                    df = add_technical_indicators(bars)
                    state = normalize_state(df, len(df)-1)
                    
                    # Check Position & Calculate P/L
                    has_position = 0.0
                    entry_price = 0.0
                    pnl_pct = 0.0
                    
                    try:
                        pos = self.api.get_position(symbol)
                        if float(pos.qty) != 0:
                            has_position = 1.0
                            entry_price = float(pos.avg_entry_price)
                            pnl_pct = (current_price - entry_price) / entry_price
                    except: pass
                    
                    portfolio_state = np.array([1.0, 0.0, 0.0, has_position, 0.0])
                    full_state = np.concatenate((state, portfolio_state))
                    
                    # ---------------------------------------------------------
                    # SMART DAY TRADING LOGIC (Hybrid AI + Quant)
                    # ---------------------------------------------------------
                    
                    # 1. Calculate Intraday VWAP
                    # Filter bars for *today* only to calc accurate day-VWAP
                    current_date = bars.index[-1].date()
                    todays_bars = bars[bars.index.date == current_date].copy()
                    
                    vwap = current_price # Fallback
                    if not todays_bars.empty:
                        todays_bars['pv'] = todays_bars['close'] * todays_bars['volume']
                        vwap = todays_bars['pv'].cumsum().iloc[-1] / todays_bars['volume'].cumsum().iloc[-1]
                    
                    # 2. Key Levels
                    days_open = float(todays_bars['open'].iloc[0]) if not todays_bars.empty else current_price
                    
                    # 3. Decision Logic
                    # We combine Model Score + Technical Factors
                    
                    # Compute Model Q-Values
                    state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(self.agent.device)
                    q_values = self.agent.model(state_tensor).detach().cpu().numpy()[0]
                    
                    # Calculate "Scalping Score"
                    # For scalping, we prioritize SPEED over perfection
                    buy_confidence = q_values[1] - q_values[0] 
                    
                    # FACTOR 1: Price vs VWAP (Still important but less strict)
                    if current_price > vwap:
                        buy_confidence += 0.20  # Bullish boost
                    else:
                        buy_confidence -= 0.15  # Minor penalty (still trade below VWAP if moving)
                        
                    # FACTOR 2: Green Day (Price > Open)
                    if current_price > days_open:
                        buy_confidence += 0.10
                    
                    # FACTOR 3: Volume Spike (Key for scalping!)
                    volume_ratio = df['volume_ratio'].iloc[-1]
                    if volume_ratio > 1.5:  # Above average volume
                        buy_confidence += 0.15  # Volume = Action
                    
                    # FACTOR 4: Momentum (Simplified - just check if moving up)
                    current_rsi = df['rsi'].iloc[-1]
                    if current_rsi > 40:  # Very liberal threshold
                        buy_confidence += 0.05
                    # REMOVED: Overbought penalty (scalpers trade momentum)
                        
                    # FINAL DECISION
                    final_action = 0 # Default Hold
                    
                    if has_position == 0:
                        # SCALPER ENTRY: Very aggressive threshold
                        # Trigger on ANY positive momentum signal
                        if buy_confidence > -0.10:  # Ultra-low bar (even slightly negative = BUY if other factors good)
                            final_action = 1 # BUY
                    
                    elif has_position == 1:
                        # =====================================================
                        # SCALPING EXIT LOGIC (Ultra-Aggressive)
                        # =====================================================
                        
                        # Rule 1: Profit Target (+0.8%)
                        if pnl_pct >= 0.008:
                            final_action = 2 # SELL - Take Profit
                        
                        # Rule 2: Stop Loss (-0.4%)
                        elif pnl_pct <= -0.004:
                            final_action = 2 # SELL - Cut Loss
                        
                        # Rule 3: VWAP Breakdown (Momentum Fail)
                        elif current_price < vwap * 0.995:
                            final_action = 2 # SELL - Trend Broken
                        
                        # Rule 4: Model SELL Signal (Backup)
                        elif q_values[2] > max(q_values[0], q_values[1]):
                            final_action = 2 # SELL - AI Signal
                    
                    unique_decisions.append(final_action)
                    
                    # Debug Info - Print scores for key symbols
                    if symbol in ['NVDA', 'TSLA', 'SPY', 'AMD', 'AAPL']:
                        print(f"   {symbol}: Confidence={buy_confidence:.3f}, Action={['HOLD', 'BUY', 'SELL'][final_action]}, Price=${current_price:.2f}, VWAP=${vwap:.2f}")


                    # Execute
                    if final_action == 1 and has_position == 0:  # BUY
                        # Check Max Positions
                        open_positions = self.api.list_positions()
                        if len(open_positions) >= 10:  # MAX_POSITIONS = 10
                            # print(f"⚠️ Max positions reached (10). Skipping BUY for {symbol}")
                            pass
                        else:
                            print(f"🟢 BUY SIGNAL: {symbol} @ ${current_price:.2f}")
                            
                            # Calculate qty based on portfolio allocation
                            acct = self.api.get_account()
                            equity = float(acct.equity)
                            buying_power = float(acct.buying_power)
                            
                            # NATURAL SIZING: Divide equity by max positions
                            MAX_POSITIONS = 10
                            target_per_position = equity / MAX_POSITIONS
                            
                            # Ensure we have enough buying power
                            allocation = min(target_per_position, buying_power * 0.90)
                            
                            qty = int(allocation / current_price)
                            
                            if qty > 0:
                                try:
                                    self.api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
                                    print(f"🚀 Executed BUY {qty} {symbol} (IEX Price: {current_price})")
                                except Exception as e:
                                    print(f"❌ Order Failed: {e}")
                            
                    elif final_action == 2 and has_position == 1: # SELL
                        print(f"🔴 SELL SIGNAL: {symbol} @ ${current_price:.2f}")
                        try:
                            self.api.close_position(symbol)
                            print(f"📉 Closed {symbol}")
                        except Exception as e:
                            print(f"❌ Close Failed: {e}")
                            
                    # Log heartbeat for active symbols
                    if symbol in ['NVDA', 'TSLA', 'SPY']:
                         print(f"   {symbol}: {['HOLD', 'BUY', 'SELL'][final_action]} (Price: ${current_price:.2f})")
                
                except Exception as e:
                    print(f"⚠️ Error {symbol}: {e}")

            # Variation check
            decisions_set = set(unique_decisions)
            print(f"✅ Scan complete. Decisions: {['HOLD', 'BUY', 'SELL'][list(decisions_set)[0]] if len(decisions_set)==1 else 'MIXED'}")
            
            time.sleep(60) # 1 Minute Interval

if __name__ == "__main__":
    bot = DayTrader()
    bot.trade_loop()
