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
    
    # ========== TREND PARAMETERS (15m) ==========
    PROFIT_TARGET = 0.008     # 0.8% Target (Quick scalps)
    STOP_LOSS = -0.003        # 0.3% Stop (Tighter control)
    TRAIL_TRIGGER = 0.006     # Start trailing at 0.6%
    TRAIL_DISTANCE = 0.003    # Trail by 0.3%
    
    MAX_POSITIONS = 10        # Focused portfolio
    POSITION_SIZE_PCT = 0.09  # ~10 positions possible
    SCAN_INTERVAL = 60        # 1 minute scan (slower pace)
    COOLDOWN_SECONDS = 600    # 10 min cooldown
    
    def __init__(self):
        self.api = tradeapi.REST(self.API_KEY, self.API_SECRET, self.BASE_URL, api_version='v2')
        self.entry_prices = {}
        self.peak_prices = {}
        self.entry_times = {}
        self.last_exit_times = {}
        
        print("="*60)
        print("⚡ CRYPTO TREND BOT (15m)")
        print("="*60)
        print(f"📊 Pairs: {len(self.CRYPTO_SYMBOLS)}")
        print(f"🎯 Target: +{self.PROFIT_TARGET*100:.1f}% | Stop: {self.STOP_LOSS*100:.1f}%")
        print(f"📈 Trail: Trigger {self.TRAIL_TRIGGER*100:.1f}% → Distance {self.TRAIL_DISTANCE*100:.1f}%")
        print("="*60)
    
    def get_bars(self, symbol, interval='15m', period='5d'):
        """Get trend bars"""
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
        """Trend Following Indicators (15m)"""
        if df is None or len(df) < 50:
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Trend Indicators
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        sma_50 = close.rolling(50).mean()
        
        # Momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        # Volatility / ATR
        # (Simplified)
        
        # Trend Status
        uptrend = (close.iloc[-1] > ema_21.iloc[-1]) and (ema_9.iloc[-1] > ema_21.iloc[-1])
        strong_uptrend = uptrend and (close.iloc[-1] > sma_50.iloc[-1])
        
        # Volume Check
        vol_avg = volume.rolling(20).mean()
        vol_strength = volume.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1
        
        return {
            'price': close.iloc[-1],
            'uptrend': uptrend,
            'strong_uptrend': strong_uptrend,
            'rsi': rsi.iloc[-1],
            'vol_strength': vol_strength,
            'ema_9': ema_9.iloc[-1],
            'ema_21': ema_21.iloc[-1]
        }

    def should_scalp_buy(self, signals):
        """Trend Following Entry Logic"""
        if signals is None:
            return False, 0
            
        score = 0
        
        # 1. MUST be in an uptrend (Price > EMA21)
        if not signals['uptrend']:
            return False, 0  # Hard reject downtrends
            
        score += 1
        
        # 2. Strong Uptrend (Above SMA50 too)
        if signals['strong_uptrend']:
            score += 2
            
        # 3. RSI Momentum (50-70 is the sweet spot for trends)
        # Avoid > 75 (Overbought) unless super strong volume
        if 50 < signals['rsi'] < 75:
            score += 2
        elif signals['rsi'] >= 75:
             # Only buy overbought if volume is massive (Climax run)
             if signals['vol_strength'] > 2.5:
                 score += 1
             else:
                 return False, 0 # risky top
                 
        # 4. Volume Support
        if signals['vol_strength'] > 1.2:
            score += 1
            
        # Threshold: conservative
        return score >= 4, score
    
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
                                
                                # Record exit time for cooldown
                                self.last_exit_times[p.symbol] = time.time()
                                
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
                        
                        # check cooldown
                        if time.time() - self.last_exit_times.get(symbol_clean, 0) < self.COOLDOWN_SECONDS:
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
                                    print(f"      Score: {score} | RSI: {sig['rsi']:.0f} | Vol: {sig['vol_strength']:.1f}x")
                                    
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
