import time
import os
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime
from typing import Optional

from config.settings import (
    CryptoTraderCreds, CryptoTraderConfig, ALPACA_BASE_URL
)
from src.strategies.supertrend import SupertrendStrategy
from src.data.market_fetcher import MarketDataFetcher

class CryptoSupertrendBot:
    """
    Live Crypto Trading Bot using the Supertrend Strategy.
    """
    def __init__(self, watchlist_path: Optional[str] = None, take_profit_pct: float = 0.03):
        print(f"🤖 Initializing Crypto Supertrend Bot...")
        
        self.creds = CryptoTraderCreds
        self.config = CryptoTraderConfig
        self.take_profit_pct = take_profit_pct
        print(f"🎯 Take Profit Target: {self.take_profit_pct*100}%")
        
        # 1. API Connection
        self.api = tradeapi.REST(
            str(self.creds.API_KEY),
            str(self.creds.API_SECRET),
            str(ALPACA_BASE_URL),
            api_version='v2'
        )
        
        # 2. Data Fetcher
        self.market_fetcher = MarketDataFetcher()
        
        # 3. Strategy
        # Standard settings: ATR 10, Multiplier 3.0
        self.strategy = SupertrendStrategy(atr_period=10, multiplier=3.0)
        
        # 4. Watchlist
        self.watchlist_path = watchlist_path
        self.symbols = self._load_watchlist()
        
        print(f"✅ Bot Ready. Watching {len(self.symbols)} symbols.")

    def _load_watchlist(self):
        path = self.watchlist_path
        if not path:
            path = "config/watchlists/crypto_watchlist.txt"
            
        if os.path.exists(path):
            with open(path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(symbols)} symbols from {path}")
            return symbols
        else:
            print(f"⚠️ Watchlist not found at {path}. Using default list.")
            return ['BTC/USD', 'ETH/USD', 'SOL/USD']

    def get_position(self, symbol):
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty), float(pos.avg_entry_price)
        except:
            return 0.0, 0.0

    def get_cash(self):
        try:
            account = self.api.get_account()
            return float(account.cash)
        except:
            return 0.0

    def execute_trade(self, symbol, signal, price):
        position_qty, _ = self.get_position(symbol)
        
        if signal == 'buy':
            if position_qty > 0:
                print(f"   ⏸️ Already holding {symbol}. Skipping Buy.")
                return

            cash = self.get_cash()
            # Simple risk management: Use 10% of available cash per trade
            # or minimum $10 (Alpaca Crypto Min), max $1000
            trade_amount = min(max(cash * 0.1, 10), 1000)
            
            if cash < trade_amount:
                print(f"   ⚠️ Insufficient cash (${cash:.2f}) for {symbol}.")
                return

            qty = trade_amount / price
            # Round down to appropriate precision (e.g. 4 decimals for crypto)
            qty = float(f"{qty:.4f}") 
            
            if qty <= 0:
                print(f"   ⚠️ Calculated quantity is 0 for {symbol}.")
                return

            print(f"   🚀 BUYING {qty} {symbol} at ${price:.2f}...")
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"   ✅ Buy Order Sent for {symbol}")
            except Exception as e:
                print(f"   ❌ Buy Failed: {e}")

        elif signal == 'sell':
            if position_qty <= 0:
                print(f"   ⏸️ No position in {symbol}. Skipping Sell.")
                return

            print(f"   📉 SELLING {position_qty} {symbol} at ${price:.2f}...")
            try:
                self.api.submit_order(
                    symbol=symbol,
                    qty=position_qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"   ✅ Sell Order Sent for {symbol}")
            except Exception as e:
                print(f"   ❌ Sell Failed: {e}")

    def run(self):
        print("\n🚀 Starting Supertrend Trading Loop (1Min)...")
        while True:
            print(f"\n⏰ Scan Time: {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in self.symbols:
                print(f"🔍 Scanning {symbol}...")
                
                # 1. Fetch Data (Need enough for ATR + Lookback)
                # 100 bars should be plenty for ATR(10) + Supertrend calculation
                df = self.market_fetcher.get_bars(symbol, lookback_days=2, timeframe='1Min')
                
                if df.empty or len(df) < 20:
                    print(f"   ⚠️ Insufficient data for {symbol}")
                    continue
                
                # 2. Calculate Strategy
                df = self.strategy.calculate_supertrend(df)
                
                # 3. Check Trend State (Enter if Bullish, Exit if Bearish)
                is_bullish = df['trend'].iloc[-1]
                current_price = df['Close'].iloc[-1]
                position_qty, avg_entry_price = self.get_position(symbol)
                
                # Calculate PnL
                pnl_pct = 0.0
                if position_qty > 0 and avg_entry_price > 0:
                    pnl_pct = (current_price - avg_entry_price) / avg_entry_price

                # Log Status
                trend_str = "BULLISH 🟢" if is_bullish else "BEARISH 🔴"
                pnl_str = f" | PnL: {pnl_pct*100:.2f}%" if position_qty > 0 else ""
                print(f"   📊 {symbol}: ${current_price:.4f} | Trend: {trend_str}{pnl_str}")
                
                if is_bullish and position_qty == 0:
                    print(f"   🚨 ENTRY SIGNAL (Trend is Bullish)")
                    self.execute_trade(symbol, 'buy', current_price)
                
                # Take Profit Logic
                elif position_qty > 0 and pnl_pct >= self.take_profit_pct:
                    print(f"   💰 TAKE PROFIT TRIGGERED: {pnl_pct*100:.2f}% >= {self.take_profit_pct*100}%")
                    self.execute_trade(symbol, 'sell', current_price)

                elif not is_bullish and position_qty > 0:
                    print(f"   🚨 EXIT SIGNAL (Trend is Bearish)")
                    self.execute_trade(symbol, 'sell', current_price)
                else:
                    if position_qty > 0:
                        print(f"   ✅ Holding {symbol} (Trend is Bullish)")
                    else:
                        print(f"   💤 Waiting for Trend Flip (Trend is Bearish)")
                
                # Sleep briefly to avoid rate limits
                time.sleep(0.5)
            
            print("💤 Sleeping for 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    bot = CryptoSupertrendBot()
    bot.run()
