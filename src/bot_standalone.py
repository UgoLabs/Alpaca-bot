"""
Standalone Paper Trading Bot - All dependencies bundled
No external utils imports to avoid alpha_vantage pollution
"""

import os
import time
import json
import pickle
from collections import deque
import pandas as pd
import numpy as np
import torch
import gym
from gym import spaces
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from datetime import datetime, timedelta
from swing_model import DuelingDQN
import yfinance as yf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

MAX_POSITIONS = 20  # Limit max positions to rank best opportunities

# =============================================================================
# TECHNICAL INDICATORS (Pure Pandas/Numpy)
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
    if 'Close' in df.columns: df['Close'] = df['Close'].replace(0, np.nan).ffill()
    if 'Volume' in df.columns: df['Volume'] = df['Volume'].replace(0, np.nan).ffill()

    close = df['Close'] if 'Close' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

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
    
    # ADX
    df['adx'] = calculate_adx(high, low, close).fillna(0)
    
    # RSI
    df['rsi'] = calculate_rsi(close).fillna(50)
    
    # Stochastic
    s_low = low.rolling(window=14).min()
    s_high = high.rolling(window=14).max()
    df['stoch_k'] = (100 * ((close - s_low) / (s_high - s_low).replace(0, 1))).fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean().fillna(50)
    
    # Williams %R
    highest_high = high.rolling(window=14).max()
    lowest_low = low.rolling(window=14).min()
    df['williams_r'] = (-100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)).fillna(-50)
    
    # Bollinger Bands
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    df['bb_mid'] = sma.fillna(0)
    df['bb_high'] = (sma + 2*std).fillna(0)
    df['bb_low'] = (sma - 2*std).fillna(0)
    df['bb_width'] = ((df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, 1)).fillna(0) * 100
    df['bb_pband'] = ((close - df['bb_low']) / (df['bb_high'] - df['bb_low']).replace(0, 1)).fillna(0.5)

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0)

    # Volume
    df['volume_sma'] = volume.rolling(window=20).mean().fillna(0)
    df['volume_ratio'] = (volume / df['volume_sma'].replace(0, 1)).fillna(1)
    
    # MFI
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, 1)
    df['mfi'] = (100 - (100 / (1 + mfr))).fillna(50)

    # Derived
    df['price_vs_sma20'] = (close - df['sma_20']) / df['sma_20'].replace(0, 1)
    df['price_vs_sma50'] = (close - df['sma_50']) / df['sma_50'].replace(0, 1)
    
    # SMA slopes
    df['sma20_slope'] = df['sma_20'].pct_change(periods=5).fillna(0)
    df['sma50_slope'] = df['sma_50'].pct_change(periods=10).fillna(0)
    
    # SMA Cross
    df['sma_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
    
    # Momentum
    df['momentum_5d'] = close.pct_change(periods=5).fillna(0)
    df['momentum_10d'] = close.pct_change(periods=10).fillna(0)
    df['momentum_20d'] = close.pct_change(periods=20).fillna(0)
    
    # Market Regime
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
    
    baseline_price = window['Close'].iloc[0] if window['Close'].iloc[0] != 0 else 1.0
    
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
    norm_sma20_slope = np.clip((window['sma_20'].pct_change(periods=5).fillna(0).values) * 10, -1, 1)
    
    latest = window.iloc[-1]
    regime_features = np.array([
        latest['regime'],                                      # -1, 0, 1
        latest['trend_strength'],                              # 0 or 1
        latest['sma_cross'],                                   # -1 or 1
        np.clip(latest['momentum_5d'] * 10, -1, 1),           # 5-day momentum
        np.clip(latest['momentum_10d'] * 5, -1, 1),           # 10-day momentum
        np.clip(latest['momentum_20d'] * 3, -1, 1),           # 20-day momentum
    ])
    
    window_features = np.column_stack((
        norm_close, norm_rsi, norm_stoch_k, norm_macd_diff, norm_bb_pband,
        norm_atr, norm_volume_ratio, norm_adx, norm_price_sma20, norm_price_sma50, norm_sma20_slope
    ))
    
    features = np.concatenate([window_features.flatten(), regime_features])
    return features.astype(np.float32)

# =============================================================================
# PAPER TRADER
# =============================================================================

class PaperTrader:
    def __init__(self, model_path='models/live_model.pth'):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        
        state_size = 231  # Model was trained with this size (verified from checkpoint)
        self.agent = DuelingDQN(state_size, 3, use_noisy=True)
        self.agent.load(model_path)
        self.agent.model.train()  # Keep in training mode for online learning
        
        self.symbols = self.load_symbols()
        self.log_file = f'logs/paper_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.model_path = model_path
        
        # Online Learning Components
        self.replay_buffer = deque(maxlen=10000)  # Store recent experiences
        self.position_states = {}  # Track entry states for open positions
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=0.0001)
        self.training_log = []
        
        # Risk Management Tracking (matches training environment)
        self.position_tracking = {}  # {symbol: {entry_price, entry_time, entry_atr, peak_price}}
        
        # Risk parameters (must match swing_environment.py)
        self.stop_loss_atr_multiplier = 2.5      # Hard stop at 2.5x ATR
        self.trailing_stop_atr_multiplier = 3.0  # Trail at 3x ATR
        self.take_profit_atr_multiplier = 4.0    # Take profit at 4x ATR
        self.rsi_extreme_overbought = 80         # Exit if RSI > 80
        self.max_hold_days = 20                  # Force exit if losing after 20 days
        
        # Load existing replay buffer if exists
        buffer_path = 'models/live_replay_buffer.pkl'
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                print(f"📚 Loaded {len(self.replay_buffer)} experiences from buffer")
            except:
                pass
        
        print(f"✅ Bot initialized with {len(self.symbols)} symbols")
        print(f"📊 Model: {model_path}")
        print(f"💰 Account: {self.base_url}")
        print(f"🧠 Online Learning: ENABLED")

    def load_symbols(self):
        paths = ['my_portfolio.txt', '../my_portfolio.txt']
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return [line.strip() for line in f if line.strip() and not line.startswith('#')]
                except: pass
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']

    def get_bars(self, symbol, timeframe='1Day', limit=300):
        """Deprecated - use get_bulk_bars_yf instead"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y', interval='1d')
            if df.empty: return None
            # Ensure columns match expected format
            df.columns = [c.capitalize() for c in df.columns]
            return df if len(df) > 0 else None
        except:
            return None
    
    def get_bulk_bars_yf(self, symbols):
        """Small-chunk batch fetching - fast and reliable"""
        import warnings
        warnings.filterwarnings('ignore')
        import logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
        print(f"   📥 Fetching {len(symbols)} symbols in small batches...")
        all_bars = {}
        chunk_size = 10  # Small chunks for reliability
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            try:
                # Batch download
                data = yf.download(
                    tickers=chunk,
                    period='1y',
                    interval='1d',
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
                
                if data.empty:
                    continue
                
                # Single symbol in chunk
                if len(chunk) == 1:
                    symbol = chunk[0]
                    df = data.copy()
                    df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                    if len(df) > 200:
                        all_bars[symbol] = df
                else:
                    # Multiple symbols - use standard MultiIndex access
                    for symbol in chunk:
                        try:
                            if symbol in data.columns.get_level_values(0):
                                df = data[symbol].copy()
                                df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                                df = df.dropna()
                                if len(df) > 200:
                                    all_bars[symbol] = df
                        except:
                            pass
                            
            except:
                pass  # Skip failed chunks
        
        print(f"   ✓ Received {len(all_bars)}/{len(symbols)} symbols")
        return all_bars

    def get_account_info(self):
        try:
            account = self.api.get_account()
            return {'equity': float(account.equity), 'cash': float(account.cash)}
        except: return {'equity': 10000, 'cash': 10000}
    
    def store_experience(self, symbol, state, action, next_state, reward):
        """Store trading experience for online learning"""
        self.replay_buffer.append((state, action, reward, next_state, False))
        
        # Save buffer periodically
        if len(self.replay_buffer) % 100 == 0:
            try:
                with open('models/live_replay_buffer.pkl', 'wb') as f:
                    pickle.dump(self.replay_buffer, f)
            except:
                pass
    
    def train_on_experiences(self, batch_size=64):
        """Train the model on collected live experiences"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])
        
        # Compute Q values
        current_q = self.agent.model(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.agent.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        # Compute loss and update
        loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def get_model_action(self, symbol):
        df = self.get_bars(symbol)
        if df is None or len(df) < 250: return 0
        
        df = add_technical_indicators(df)
        current_step = len(df) - 1
        
        market_state = normalize_state(df, current_step)
        account = self.get_account_info()
        
        market_value = 0
        try:
            position = self.api.get_position(symbol)
            market_value = float(position.market_value)
        except: pass
        
        portfolio_state = np.array([
            account['cash'] / account['equity'],
            market_value / account['equity'],
            0.0,
            1.0 if market_value > 0 else 0.0,
            0.0
        ])
        
        state = np.concatenate((market_state, portfolio_state))
        action = self.agent.act(state, epsilon=0.0)
        
        return action

    def check_risk_stops(self, symbol, df, position):
        """
        Check if position should be closed due to risk management rules.
        Returns: (should_close, reason)
        """
        if symbol not in self.position_tracking:
            # Initialize tracking for existing position
            current_price = float(df['Close'].iloc[-1])
            self.position_tracking[symbol] = {
                'entry_price': float(position.avg_entry_price),
                'entry_time': datetime.now(),  # Approximate
                'entry_atr': df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02,
                'peak_price': current_price
            }
            return False, None
        
        tracking = self.position_tracking[symbol]
        current_price = float(df['Close'].iloc[-1])
        entry_price = tracking['entry_price']
        entry_atr = tracking['entry_atr']
        
        # Update peak price
        if current_price > tracking['peak_price']:
            tracking['peak_price'] = current_price
        
        # 1. Hard Stop Loss (2.5x ATR below entry)
        hard_stop = entry_price - (entry_atr * self.stop_loss_atr_multiplier)
        if current_price < hard_stop:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return True, f"HARD_STOP ({pnl_pct:+.2f}%)"
        
        # 2. Take Profit (4x ATR above entry, or 6x if ADX > 30)
        adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
        multiplier = 6.0 if adx > 30 else self.take_profit_atr_multiplier
        take_profit = entry_price + (entry_atr * multiplier)
        if current_price >= take_profit:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return True, f"TAKE_PROFIT ({pnl_pct:+.2f}%)"
        
        # 3. RSI Extreme (RSI > 80 and in profit)
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        if rsi > self.rsi_extreme_overbought and current_price > entry_price:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return True, f"RSI_EXTREME ({pnl_pct:+.2f}%)"
        
        # 4. Trailing Stop (3x ATR below peak, only if in profit)
        if current_price > entry_price:
            trailing_stop = tracking['peak_price'] - (entry_atr * self.trailing_stop_atr_multiplier)
            if current_price < trailing_stop:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                return True, f"TRAILING_STOP ({pnl_pct:+.2f}%)"
        
        # 5. Time Stop (losing after 20 days)
        days_held = (datetime.now() - tracking['entry_time']).days
        if days_held > self.max_hold_days and current_price < entry_price:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            return True, f"TIME_STOP ({pnl_pct:+.2f}%)"
        
        return False, None

    def run_once(self):
        print(f"\n{'='*60}")
        print(f"🔄 Scanning {len(self.symbols)} symbols...")
        print(f"{'='*60}\n")
        
        # BATCH FETCH ALL DATA
        all_data = self.get_bulk_bars_yf(self.symbols)
        
        if not all_data:
            print("⚠️ No data received from yfinance")
            return
            
        # Bulk fetch all positions once
        try:
            alpaca_positions = {p.symbol: p for p in self.api.list_positions()}
        except:
            alpaca_positions = {}
            
        print(f"   ℹ️  Cached {len(alpaca_positions)} open positions")
        
        potential_buys = []  # Store (symbol, price, confidence, state, qty)
        
        for symbol in self.symbols:
            try:
                # Get data from batch
                df = all_data.get(symbol)
                if df is None or len(df) < 250:
                    continue
                
                # Calculate indicators
                df = add_technical_indicators(df)
                current_step = len(df) - 1
                market_state = normalize_state(df, current_step)
                
                account = self.get_account_info()
                
                market_value = 0
                has_position = False
                
                if symbol in alpaca_positions:
                    p = alpaca_positions[symbol]
                    market_value = float(p.market_value)
                    has_position = float(p.qty) > 0
                    
                    # CHECK RISK STOPS FIRST (overrides AI decision)
                    if has_position:
                        should_close, stop_reason = self.check_risk_stops(symbol, df, p)
                        if should_close:
                            try:
                                self.api.close_position(symbol)
                                print(f"{symbol:6s} → 🛑 {stop_reason}")
                                # Clean up tracking
                                if symbol in self.position_tracking:
                                    del self.position_tracking[symbol]
                                if symbol in self.position_states:
                                    del self.position_states[symbol]
                                continue  # Skip AI decision for this symbol
                            except Exception as e:
                                print(f"{symbol:6s} → ❌ STOP FAILED: {str(e)[:30]}")
                
                portfolio_state = np.array([
                    account['cash'] / account['equity'],
                    market_value / account['equity'],
                    0.0,
                    1.0 if market_value > 0 else 0.0,
                    0.0
                ])
                
                state = np.concatenate((market_state, portfolio_state))
                
                # Get Action and Confidence from Model
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.agent.model(state_tensor)
                    action = q_values.argmax(1).item()
                    confidence = q_values[0][action].item()
                
                action_name = ['HOLD', 'BUY', 'SELL'][action]
                
                # Execute Sells Immediately
                if action == 2 and has_position:  # SELL
                    try:
                        p = alpaca_positions[symbol]
                        entry_price = float(p.avg_entry_price)
                        current_price = float(df['Close'].iloc[-1])
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        self.api.close_position(symbol)
                        
                        if symbol in self.position_states:
                            entry_state = self.position_states[symbol]['state']
                            reward = pnl_pct * 100
                            self.store_experience(symbol, entry_state, 1, state, reward)
                            del self.position_states[symbol]
                            print(f"{symbol:6s} → 🔴 SOLD (P/L: {pnl_pct*100:+.2f}%) [Learned ✓]")
                        else:
                            print(f"{symbol:6s} → 🔴 SOLD (P/L: {pnl_pct*100:+.2f}%)")
                    except Exception as e:
                        print(f"{symbol:6s} → ❌ SELL FAILED: {str(e)[:30]}")

                # Collect Buys for Ranking
                elif action == 1 and not has_position:  # BUY
                    current_price = float(df['Close'].iloc[-1])
                    allocation = account['equity'] * 0.05
                    qty = int(allocation / current_price)
                    if qty > 0:
                        potential_buys.append({
                            'symbol': symbol,
                            'price': current_price,
                            'confidence': confidence,
                            'state': state,
                            'qty': qty,
                            'atr': df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                        })
                    else:
                        print(f"{symbol:6s} → HOLD (qty 0)")

                else:
                    print(f"{symbol:6s} → {action_name} (conf: {confidence:.2f})")
                    
            except Exception as e:
                print(f"{symbol:6s} → ERROR: {str(e)[:30]}")

        # Execute Top Ranked Buys
        if potential_buys:
            # Sort by confidence
            potential_buys.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Check slots
            current_positions = len(self.api.list_positions())
            slots_available = MAX_POSITIONS - current_positions
            
            if slots_available <= 0:
                print(f"\n⚠️ Max positions ({MAX_POSITIONS}) reached. Skipping buys.")
            else:
                top_picks = potential_buys[:slots_available]
                print(f"\n🎯 Processing top {len(top_picks)} buys from {len(potential_buys)} candidates:")
                
                for pick in top_picks:
                    symbol = pick['symbol']
                    try:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=pick['qty'],
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        # Track state/risk
                        self.position_states[symbol] = {
                            'state': pick['state'],
                            'entry_price': pick['price'],
                            'entry_time': datetime.now()
                        }
                        self.position_tracking[symbol] = {
                            'entry_price': pick['price'],
                            'entry_time': datetime.now(),
                            'entry_atr': pick['atr'],
                            'peak_price': pick['price']
                        }
                        print(f"{symbol:6s} → 🟢 BUY {pick['qty']} @ ${pick['price']:.2f} (conf: {pick['confidence']:.3f})")
                    except Exception as e:
                        print(f"{symbol:6s} → ❌ BUY FAILED: {str(e)[:30]}")
        
        # Online Learning: Train on experiences 
        if len(self.replay_buffer) >= 64:
            # Train every scan if we have enough data
            loss = self.train_on_experiences(batch_size=64)
            if loss > 0:
                print(f"\n🧠 Trained on {len(self.replay_buffer)} experiences (Loss: {loss:.4f})")
                
                # Save improved model every 5 scans
                if not hasattr(self, 'scan_count'):
                    self.scan_count = 0
                self.scan_count += 1
                
                if self.scan_count % 5 == 0:
                    self.agent.save(self.model_path)
                    print(f"💾 Saved improved model to {self.model_path}")
        
        print(f"\n{'='*60}")
        print(f"✅ Scan complete at {datetime.now().strftime('%H:%M:%S')}")
        print(f"📚 Replay Buffer: {len(self.replay_buffer)} experiences")
        print(f"{'='*60}\n")

    def run_loop(self, interval_minutes=15):
        print(f"🚀 Starting continuous trading (interval: {interval_minutes}min)")
        import pytz
        eastern = pytz.timezone('US/Eastern')
        
        while True:
            try:
                now_et = datetime.now(eastern)
                today_str = now_et.strftime('%Y-%m-%d')
                
                # Default timings (fallback)
                market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                
                try:
                    # Get correct schedule (Handles Holidays/Early Closes)
                    schedules = self.api.get_calendar(start=today_str, end=today_str)
                    if not schedules:
                        print("💤 Market Closed (Holiday) - Sleeping 1 hour")
                        time.sleep(3600)
                        continue
                        
                    s = schedules[0]
                    # Alpaca returns datetime objects or time objects
                    if hasattr(s.open, 'astimezone'):
                        market_open = s.open.astimezone(eastern)
                        market_close = s.close.astimezone(eastern)
                    else:
                        # s.open and s.close are time objects, assume they're in ET already
                        market_open = eastern.localize(datetime.combine(now_et.date(), s.open))
                        market_close = eastern.localize(datetime.combine(now_et.date(), s.close))
                except Exception as e:
                    print(f"⚠️ Calendar Check Failed: {e}. Using Default 9:30-16:00 ET.")

                # Check if market is open
                if now_et < market_open:
                    time_to_open = (market_open - now_et).total_seconds() / 60
                    print(f"💤 Pre-Market. Opening in {time_to_open:.0f} mins at {market_open.strftime('%H:%M')} ET")
                    time.sleep(min(300, time_to_open * 60))  # Sleep up to 5 mins
                    continue
                elif now_et > market_close:
                    print(f"💤 Market Closed (Closed at {market_close.strftime('%H:%M')} ET)")
                    time.sleep(3600)  # Sleep 1 hour
                    continue
                
                # Market is open - run trading
                self.run_once()
                print(f"⏳ Sleeping {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone Paper Trading Bot')
    parser.add_argument('--model', type=str, default='models/SHARED_dqn_best.pth')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=15, help='Loop interval (minutes)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 ALPACA TRADING BOT - STANDALONE DEPLOYMENT")
    print("="*60 + "\n")
    
    trader = PaperTrader(model_path=args.model)
    
    if args.once:
        trader.run_once()
    else:
        trader.run_loop(interval_minutes=args.interval)
