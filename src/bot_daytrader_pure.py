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

PROFIT_TARGET = 0.008  # 0.8%
STOP_LOSS = 0.004      # 0.4%
MAX_POSITIONS = 10

class PureDayTrader:
    def __init__(self):
        self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        self.symbols = self.load_watchlist()
        print(f"🤖 Pure Rule-Based Day Trader (YFinance Data)")
        print(f"📋 Watchlist: {len(self.symbols)} symbols")
        print(f"🎯 Target: +{PROFIT_TARGET*100}% | Stop: -{STOP_LOSS*100}%")

    def load_watchlist(self):
        try:
            with open('../my_portfolio.txt', 'r') as f:
                symbols = [l.strip() for l in f if l.strip()]
                print(f"📋 Loaded {len(symbols)} symbols from portfolio")
                return symbols
        except:
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
            # 1. Market Open Check & EOD Liquidation
            # 1. Market Open Check & EOD Liquidation
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    now = datetime.now(clock.timestamp.tzinfo)
                    next_open = clock.next_open
                    time_to_open = (next_open - now).total_seconds()
                    
                    if time_to_open > 300:
                        sleep_time = time_to_open - 300  # Wake up 5 mins early
                        wake_time = now.timestamp() + sleep_time
                        wake_dt = datetime.fromtimestamp(wake_time)
                        
                        print(f"💤 Market closed. Sleeping until {wake_dt.strftime('%H:%M:%S')} (5 min before open)")
                        time.sleep(sleep_time)
                        
                        # Refresh clock after waking up
                        clock = self.api.get_clock()
                        if not clock.is_open:
                            print("⏳ Waking up... Waiting for market open...")
                            time.sleep(60)
                            continue
                    else:
                         print("💤 Market closed")
                         time.sleep(60)
                         continue
                
                # EOD Liquidation: Close all positions 15 min before close
                import pytz
                eastern = pytz.timezone('US/Eastern')
                now_et = datetime.now(eastern)
                market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                time_to_close = (market_close - now_et).total_seconds() / 60  # minutes
                
                if time_to_close <= 15 and time_to_close > 0:
                    positions = self.api.list_positions()
                    if positions:
                        print(f"\n🔔 EOD LIQUIDATION: Closing {len(positions)} positions...")
                        for p in positions:
                            try:
                                self.api.close_position(p.symbol)
                                print(f"   📤 Closed {p.symbol}")
                            except:
                                pass
                        print("✅ All positions liquidated for EOD")
                        time.sleep(900)  # Wait 15 min until market close
                        continue
                        
            except: pass
            
            # 2. Check Exits
            try:
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
                            elif pnl <= -STOP_LOSS:
                                self.api.close_position(p.symbol)
                                print(f"   📉 CLOSE {p.symbol}: STOP LOSS ({pnl*100:.2f}%)")
                        except: pass
            except: pass

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
                    
                    if score >= 3 or symbol in ['NVDA', 'TSLA', 'SPY']:
                        print(f" {status} {symbol:<5} | Score: {score}/7 | ${current_price:<8.2f} | {', '.join(triggers)}")

                    # Execute
                    if score >= 4:
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
            time.sleep(30)  # Fast!

if __name__ == "__main__":
    bot = PureDayTrader()
    bot.run()
