"""
Crypto Scalping Bot - Paper Account 3
Ultra-tight entries/exits for quick profits on high-volume crypto
"""

import os
import time
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CryptoScalper:
    # ========== PAPER ACCOUNT 3 CREDENTIALS ==========
    API_KEY = 'PKPTQHGKGTKXFYQHMKJXIULCYF'
    API_SECRET = '8Z5fpbuCA9xcjt4s5aAwiLfarR6sW2XBf9aqswKvjQwK'
    BASE_URL = 'https://paper-api.alpaca.markets'
    
    # Focus on high-liquidity pairs (yfinance compatible)
    CRYPTO_SYMBOLS = [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'DOGE/USD',
        'AVAX/USD', 'LINK/USD', 'LTC/USD', 'BCH/USD', 'XLM/USD',
        'DOT/USD', 'ATOM/USD', 'AAVE/USD', 'SHIB/USD', 'ALGO/USD'
    ]
    
    # ========== SCALPING PARAMETERS ==========
    PROFIT_TARGET = 0.003     # 0.3% Take Profit (Very fast execution)
    STOP_LOSS = -0.002        # Stop Loss -0.2% (Tighter!)
    TRAIL_TRIGGER = 0.002     # Trail starts at +0.2%
    TRAIL_DISTANCE = 0.001    # Trail distance 0.1%
    
    MAX_POSITIONS = 20        # Virtually unlimited (covers all 15 symbols)
    POSITION_SIZE_PCT = 0.065 # 6.5% size allows holding all ~15 symbols (15 * 6.5% ~= 97.5%)
    SCAN_INTERVAL = 10        # 10s Loop
    
    def __init__(self):
        self.api = tradeapi.REST(self.API_KEY, self.API_SECRET, self.BASE_URL, api_version='v2')
        self.entry_prices = {}  # Track entry for trailing stops
        self.peak_prices = {}   # Track peak for trailing
        self.entry_times = {}   # Track entry time for stale exit
        
        print("="*60)
        print("⚡ CRYPTO SCALPING BOT")
        print("="*60)
        print(f"📊 Pairs: {len(self.CRYPTO_SYMBOLS)}")
        print(f"🎯 Target: +{self.PROFIT_TARGET*100:.1f}% | Stop: {self.STOP_LOSS*100:.1f}%")
        print(f"📈 Trail: Trigger {self.TRAIL_TRIGGER*100:.1f}% → Distance {self.TRAIL_DISTANCE*100:.1f}%")
        print(f"⏱️  Scan: Every {self.SCAN_INTERVAL}s")
        print("="*60)
    
    def get_bars(self, symbol, interval='5m', period='1d'):
        """Get short-term bars for scalping"""
        try:
            yf_symbol = symbol.replace('/', '-')
            data = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                return None
            
            # Handle MultiIndex columns (new yfinance format)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Normalize column names to lowercase
            data.columns = [str(c).lower() for c in data.columns]
            
            # Ensure we have required columns
            if 'close' not in data.columns:
                return None
                
            return data
        except Exception as e:
            return None
    
    def calculate_scalp_signals(self, df):
        """Fast scalping indicators"""
        if df is None or len(df) < 20:
            return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Fast EMAs for quick signals
        ema_5 = close.ewm(span=5).mean()
        ema_13 = close.ewm(span=13).mean()
        
        # RSI (short period for scalping)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(7).mean()
        loss = -delta.where(delta < 0, 0).rolling(7).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic RSI for overbought/oversold
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1)
        
        # Volume spike detection
        vol_sma = volume.rolling(10).mean()
        vol_spike = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1
        
        # Price momentum (last 3 candles)
        momentum = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100
        
        # Micro trend (last 5 candles)
        micro_trend = "UP" if close.iloc[-1] > close.iloc[-5] > close.iloc[-10] else \
                      "DOWN" if close.iloc[-1] < close.iloc[-5] < close.iloc[-10] else "FLAT"
        
        # Candle strength (current candle body vs range)
        body = abs(close.iloc[-1] - df['open'].iloc[-1])
        range_size = high.iloc[-1] - low.iloc[-1]
        candle_strength = body / range_size if range_size > 0 else 0
        
        # Green candle?
        is_green = close.iloc[-1] > df['open'].iloc[-1]
        
        return {
            'price': close.iloc[-1],
            'ema_5': ema_5.iloc[-1],
            'ema_13': ema_13.iloc[-1],
            'ema_cross_up': ema_5.iloc[-1] > ema_13.iloc[-1] and ema_5.iloc[-2] <= ema_13.iloc[-2],
            'ema_bullish': ema_5.iloc[-1] > ema_13.iloc[-1],
            'rsi': rsi.iloc[-1],
            'stoch_rsi': stoch_rsi.iloc[-1],
            'vol_spike': vol_spike,
            'momentum': momentum,
            'micro_trend': micro_trend,
            'candle_strength': candle_strength,
            'is_green': is_green
        }
    
    def should_scalp_buy(self, signals):
        """Scalping entry conditions - need multiple confirmations"""
        if signals is None:
            return False, 0
        
        score = 0
        
        # EMA crossover (strongest signal)
        if signals['ema_cross_up']:
            score += 3
        elif signals['ema_bullish']:
            score += 1
        
        # RSI in buy zone (not overbought)
        if 35 < signals['rsi'] < 65:
            score += 1
        
        # Stochastic RSI oversold bounce
        if signals['stoch_rsi'] < 0.3:
            score += 2
        elif signals['stoch_rsi'] < 0.5:
            score += 1
        
        # Volume spike (confirms move)
        if signals['vol_spike'] > 2.0:
            score += 2
        elif signals['vol_spike'] > 1.5:
            score += 1
        
        # Positive momentum
        if signals['momentum'] > 0.2:
            score += 1
        
        # Strong green candle
        if signals['is_green'] and signals['candle_strength'] > 0.6:
            score += 1
        
        # Micro trend alignment
        if signals['micro_trend'] == 'UP':
            score += 1
        
        # Require high confidence for scalping
        return score >= 5, score
    
    def manage_position(self, position, current_price):
        """Trailing stop management for scalping"""
        symbol = position.symbol
        entry = float(position.avg_entry_price)
        pnl_pct = (current_price - entry) / entry
        
        # Initialize tracking
        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = entry
            self.peak_prices[symbol] = current_price
            self.entry_times[symbol] = time.time()
        
        # Update peak
        if current_price > self.peak_prices[symbol]:
            self.peak_prices[symbol] = current_price
        
        peak = self.peak_prices[symbol]
        drawdown_from_peak = (current_price - peak) / peak
        
        # Calculate duration
        duration_mins = (time.time() - self.entry_times.get(symbol, time.time())) / 60
        
        # Exit conditions
        action = None
        reason = None
        
        # 1. Hard stop loss
        if pnl_pct <= self.STOP_LOSS:
            action = 'STOP'
            reason = f'Stop Loss {pnl_pct*100:.2f}%'
        
        # 2. Take profit target
        elif pnl_pct >= self.PROFIT_TARGET:
            action = 'PROFIT'
            reason = f'Target Hit {pnl_pct*100:.2f}%'
        
        # 3. Trailing stop (if we've reached trigger)
        elif pnl_pct >= self.TRAIL_TRIGGER and drawdown_from_peak <= -self.TRAIL_DISTANCE:
            action = 'TRAIL'
            reason = f'Trail Stop {pnl_pct*100:.2f}% (peak: {((peak-entry)/entry)*100:.2f}%)'
            
        # 4. Stale Exit (Time limit) - if held > 20 mins and slightly green, just take it.
        elif duration_mins > 20 and pnl_pct > 0.001:
            action = 'STALE'
            reason = f'Stale Exit (+{pnl_pct*100:.2f}%)'
        
        return action, reason, pnl_pct
    
    def run(self):
        print(f"\n🚀 Starting 24/7 crypto scalping...\n")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                now = datetime.now().strftime('%H:%M:%S')
                
                # ===== MANAGE EXISTING POSITIONS =====
                positions = self.api.list_positions()
                
                if positions:
                    print(f"[{now}] 📊 Managing {len(positions)} positions...")
                    
                    for p in positions:
                        try:
                            current_price = float(p.current_price)
                            action, reason, pnl = self.manage_position(p, current_price)
                            
                            if action:
                                self.api.close_position(p.symbol)
                                emoji = '✅' if action == 'PROFIT' else '🛑' if action == 'STOP' else '📈'
                                print(f"   {emoji} {p.symbol}: {reason}")
                                
                                # Cleanup tracking
                                if p.symbol in self.entry_prices:
                                    del self.entry_prices[p.symbol]
                                if p.symbol in self.peak_prices:
                                    del self.peak_prices[p.symbol]
                            else:
                                print(f"   💎 {p.symbol}: {pnl*100:+.2f}% (holding)")
                                
                        except Exception as e:
                            print(f"   ⚠️ {p.symbol}: Error - {str(e)[:20]}")
                
                # ===== SCAN FOR NEW ENTRIES =====
                if len(positions) < self.MAX_POSITIONS:
                    # Convert position symbols to comparable format (remove slashes)
                    current_symbols = [p.symbol.replace('/', '') for p in positions]
                    opportunities = []
                    
                    for symbol in self.CRYPTO_SYMBOLS:
                        # Skip if we already own this
                        symbol_clean = symbol.replace('/', '')
                        if symbol_clean in current_symbols:
                            continue
                        
                        df = self.get_bars(symbol, interval='5m', period='1d')
                        signals = self.calculate_scalp_signals(df)
                        should_buy, score = self.should_scalp_buy(signals)
                        
                        if should_buy:
                            opportunities.append((symbol, score, signals['price'], signals))
                    
                    if opportunities:
                        opportunities.sort(key=lambda x: x[1], reverse=True)
                        print(f"[{now}] 🔎 Found {len(opportunities)} opportunities")
                        
                        slots = self.MAX_POSITIONS - len(positions)
                        for symbol, score, price, sig in opportunities[:slots]:
                            try:
                                account = self.api.get_account()
                                equity = float(account.equity)
                                
                                allocation = equity * self.POSITION_SIZE_PCT
                                qty = allocation / price
                                
                                # Crypto quantity precision
                                if 'BTC' in symbol:
                                    qty = round(qty, 4)
                                elif 'ETH' in symbol:
                                    qty = round(qty, 3)
                                else:
                                    qty = round(qty, 2)
                                
                                if qty > 0:
                                    self.api.submit_order(
                                        symbol=symbol,
                                        qty=qty,
                                        side='buy',
                                        type='market',
                                        time_in_force='gtc'
                                    )
                                    print(f"   ⚡ SCALP {symbol}: {qty} @ ${price:.4f}")
                                    print(f"      Score: {score} | RSI: {sig['rsi']:.0f} | Vol: {sig['vol_spike']:.1f}x")
                                    
                            except Exception as e:
                                print(f"   ❌ {symbol}: {str(e)[:30]}")
                    else:
                        if scan_count % 20 == 0:  # Print every ~5 min
                            print(f"[{now}] 👀 No scalp setups found...")
                
                time.sleep(self.SCAN_INTERVAL)
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    scalper = CryptoScalper()
    scalper.run()
