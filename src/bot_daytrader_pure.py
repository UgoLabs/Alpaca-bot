"""
Pure Rule-Based Day Trader
Uses yfinance for data (unlimited) and Alpaca for trading
"""

import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import datetime, timedelta
import pytz

# Configuration
API_KEY = "PKA7TFQVG5OB3YK6UEJ6ZFEGOH"
API_SECRET = "6ceJ8ZhknodD8iGM2NuMYTpxjr4BMgc5DaoD1xCagtbp"
BASE_URL = "https://paper-api.alpaca.markets"

PROFIT_TARGET = 0.005  # 0.5% (Quick scalps)
STOP_LOSS = 0.003      # 0.3% (Tight control)
MAX_POSITIONS = 10
COOLDOWN_SECONDS = 300  # 5 min cooldown after exit

class PureDayTrader:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        self.symbols = self.load_watchlist()
        self.last_exit_times = {}  # Track exits for cooldown
        print(f"🤖 Pure Rule-Based Day Trader (YFinance Data)")
        print(f"📋 Watchlist: {len(self.symbols)} symbols")
        print(f"🎯 Target: +{PROFIT_TARGET*100}% | Stop: -{STOP_LOSS*100}%")

    def load_watchlist(self):
        try:
            paths = ['my_portfolio.txt', '../my_portfolio.txt']
            for path in paths:
                try:
                    with open(path, 'r') as f:
                        symbols = [l.strip() for l in f if l.strip()]
                        print(f"📋 Loaded {len(symbols)} symbols from {path}")
                        return symbols
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
            print("⚠️ Portfolio file not found, using defaults")
            return ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'SPY', 'QQQ', 'META', 'AMZN']
        except Exception as e:
            print(f"Global load error: {e}")
            return ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'SPY', 'QQQ', 'META', 'AMZN']

    def get_bulk_data_yf(self):
        """Fetch 15min bars using yfinance with chunking for reliability"""
        try:
            print(f"   Fetching {len(self.symbols)} symbols via yfinance...")
            
            all_bars = {}
            chunk_size = 20  # Process in chunks for reliability
            
            for i in range(0, len(self.symbols), chunk_size):
                chunk = self.symbols[i:i+chunk_size]
                
                try:
                    # Download batch data for chunk
                    data = yf.download(
                        tickers=chunk,
                        period='5d',
                        interval='15m',
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
                        df.columns = df.columns.str.lower()
                        if len(df) >= 50:
                            all_bars[symbol] = df
                    else:
                        # Multiple symbols
                        for symbol in chunk:
                            try:
                                if symbol in data.columns.get_level_values(0):
                                    df = data[symbol].copy()
                                    df.columns = df.columns.str.lower()
                                    df = df.dropna()
                                    if len(df) >= 50:
                                        all_bars[symbol] = df
                            except:
                                pass
                except:
                    pass  # Skip failed chunks
            
            print(f"   ✓ Received {len(all_bars)} symbols")
            return all_bars
            
        except Exception as e:
            print(f"   ⚠️ YFinance error: {e}")
            return {}

    def calculate_signals(self, df):
        """Calculate day trading signals"""
        if df.empty: return None
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # VWAP (Today only)
        current_date = df.index[-1].date()
        today_df = df[df.index.date == current_date].copy()
        
        if len(today_df) > 0:
            today_df['pv'] = today_df['close'] * today_df['volume']
            vwap = today_df['pv'].cumsum().iloc[-1] / today_df['volume'].cumsum().iloc[-1]
            days_open = today_df['open'].iloc[0]
        else:
            vwap = close.iloc[-1]
            days_open = close.iloc[-1]
        
        # EMAs
        ema_9 = close.ewm(span=9).mean()
        ema_21 = close.ewm(span=21).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        # Volume
        vol_sma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1]
        
        return {
            'price': close.iloc[-1],
            'vwap': vwap,
            'days_open': days_open,
            'ema9': ema_9.iloc[-1],
            'ema21': ema_21.iloc[-1],
            'rsi': rsi.iloc[-1],
            'vol_ratio': vol_ratio
        }

    def run(self):
        print("⏱️  Scanning every 30 seconds (YFinance - No Limits)...")
        
        while True:
            # 1. Market Schedule & EOD Liquidation
            try:
                import pytz
                eastern = pytz.timezone('US/Eastern')
                now_et = datetime.now(eastern)
                today_str = now_et.strftime('%Y-%m-%d')
                
                # Default timings (fallback)
                market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                
                try:
                    # Get correct schedule (Handles Holidays/Early Closes)
                    schedules = self.api.get_calendar(start=today_str, end=today_str)
                    if not schedules:
                        print("💤 Market Closed (Holiday)")
                        time.sleep(3600)
                        continue
                        
                    s = schedules[0]
                    # Alpaca returns datetime objects - they should already be timezone-aware
                    # If they're datetime.time objects, combine with today's date
                    if hasattr(s.open, 'astimezone'):
                        market_open = s.open.astimezone(eastern)
                        market_close = s.close.astimezone(eastern)
                    else:
                        # s.open and s.close are time objects, assume they're in ET already
                        market_open = eastern.localize(datetime.combine(now_et.date(), s.open))
                        market_close = eastern.localize(datetime.combine(now_et.date(), s.close))
                except Exception as e:
                    print(f"⚠️ Calendar Check Failed: {e}. Using Default 9:30-16:00 ET.")

                # Strict Status Check
                if now_et < market_open:
                     print(f"💤 Pre-Market. Waiting for {market_open.strftime('%H:%M')} ET")
                     time.sleep(60)
                     continue
                elif now_et > market_close:
                     print(f"💤 Market Closed (Closed at {market_close.strftime('%H:%M')} ET)")
                     time.sleep(60)
                     continue
                
                # EOD Liquidation (15 mins before close)
                time_to_close = (market_close - now_et).total_seconds() / 60
                if time_to_close <= 15 and time_to_close > 0:
                     print("🔔 EOD LIQUIDATION initiated...")
                     positions = self.api.list_positions()
                     for p in positions:
                         try:
                             self.api.close_position(p.symbol)
                             print(f"   📤 Closed {p.symbol}")
                         except: pass
                     print("✅ Liquidation Complete. Waiting for close.")
                     time.sleep(900)
                     continue

                # 2. Manage Positions (Exits)
                positions = self.api.list_positions()
                if positions:
                    print(f"\n🔍 Checking {len(positions)} positions...")
                    for p in positions:
                        try:
                            trade = self.api.get_latest_trade(p.symbol)
                            current_price = float(trade.price)
                            entry = float(p.avg_entry_price)
                            pnl = (current_price - entry) / entry
                            
                            if pnl >= PROFIT_TARGET:
                                self.api.close_position(p.symbol)
                                print(f"   📉 CLOSE {p.symbol}: PROFIT (+{pnl*100:.2f}%)")
                                self.last_exit_times[p.symbol] = time.time()  # Record exit
                            elif pnl <= -STOP_LOSS:
                                self.api.close_position(p.symbol)
                                print(f"   📉 CLOSE {p.symbol}: STOP LOSS ({pnl*100:.2f}%)")
                                self.last_exit_times[p.symbol] = time.time()  # Record exit
                        except: pass
            except Exception as e:
                print(f"Loop Error: {e}")

            # 3. FETCH DATA (yfinance)
            print("\n📥 Fetching market data...")
            data_map = self.get_bulk_data_yf()
            
            if not data_map:
                print("   ⚠️ No data, retrying in 10s...")
                time.sleep(10)
                continue
            
            # 4. Process Signals
            for symbol, bars in data_map.items():
                try:
                    signals = self.calculate_signals(bars)
                    if not signals: continue
                    
                    current_price = signals['price']

                    # Scoring
                    score = 0
                    triggers = []
                    
                    if current_price > signals['vwap']:
                        score += 2
                        triggers.append("VWAP")
                    if signals['ema9'] > signals['ema21']:
                        score += 2
                        triggers.append("EMA")
                    if 40 < signals['rsi'] < 70:
                        score += 1
                        triggers.append("RSI")
                    if signals['vol_ratio'] > 1.3:
                        score += 1
                        triggers.append("VOL")
                    if current_price > signals['days_open']:
                        score += 1
                        triggers.append("GREEN")

                    # Log
                    status = "⚪"
                    if score >= 4: status = "🟢 BUY"
                    
                    if score >= 4 or symbol in ['NVDA', 'TSLA', 'SPY']:
                        print(f" {status} {symbol:<5} | Score: {score}/7 | ${current_price:<8.2f} | {', '.join(triggers)}")

                    # Execute (Stricter criteria)
                    if score >= 4:
                        # Check cooldown
                        if time.time() - self.last_exit_times.get(symbol, 0) < COOLDOWN_SECONDS:
                            continue
                        
                        positions = self.api.list_positions()
                        is_holding = any(p.symbol == symbol for p in positions)
                        
                        if is_holding: continue

                        if len(positions) >= MAX_POSITIONS:
                            print(f"   ⚠️ Max positions reached")
                            continue
                            
                        acct = self.api.get_account()
                        equity = float(acct.equity)
                        target = equity / MAX_POSITIONS
                        qty = int(target / current_price)
                        
                        if qty > 0:
                            try:
                                self.api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
                                print(f"   🚀 EXECUTED BUY: {qty} {symbol} @ ${current_price:.2f}")
                            except Exception as e:
                                print(f"   ❌ ORDER FAILED: {e}")
                
                except Exception as e:
                    print(f"⚠️ Error {symbol}: {e}")
            
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Scan complete")
            time.sleep(30)  # Keep fast scanning

if __name__ == "__main__":
    bot = PureDayTrader()
    bot.run()
