import time
import os
import argparse
import torch
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import (
    DayTraderCreds, SwingTraderCreds, MoneyScraperCreds, CryptoTraderCreds,
    DayTraderConfig, SwingTraderConfig, MoneyScraperConfig, CryptoTraderConfig,
    ALPACA_BASE_URL, SWING_MODEL_PATH, SCALPER_MODEL_PATH, SHARED_MODEL_PATH
)
from src.agents.ensemble_agent import EnsembleAgent
from src.data.pipeline import MultiModalDataPipeline
from src.data.websocket_stream import AlpacaWebSocketStream
from src.strategies.supertrend import SupertrendStrategy  # <--- Added


class MultiModalTrader:
    """
    Live Trading Bot using the Multi-Modal Agent (Transformer + Vision + Text).
    """
    def __init__(self, watchlist_path: Optional[str] = None, mode: str = "swing"):
        print(f"🤖 Initializing Multi-Modal Trader ({mode.upper()} Mode)...")
        
        self.mode = mode.lower()
        self.watchlist_path = watchlist_path
        
        # Select Credentials and Config based on mode
        if self.mode == 'swing':
            self.creds = SwingTraderCreds
            self.config = SwingTraderConfig
        elif self.mode == 'day':
            self.creds = DayTraderCreds
            self.config = DayTraderConfig
        elif self.mode == 'scraper':
            self.creds = MoneyScraperCreds
            self.config = MoneyScraperConfig
        elif self.mode == 'crypto':
            self.creds = CryptoTraderCreds
            self.config = CryptoTraderConfig
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 1. API Connection
        self.api = tradeapi.REST(
            str(self.creds.API_KEY),
            str(self.creds.API_SECRET),
            str(ALPACA_BASE_URL),
            api_version='v2'
        )
        
        # 2. Configuration
        self.symbols = self._load_watchlist()
        self.window_size = 60
        self.num_features = 11 # OHLCV + 6 Indicators
        self.vision_channels = 11 # Same as features for 1D ResNet
        self.action_dim = 3 # Hold, Buy, Sell
        
        # 3. Initialize Brain (Agent)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Brain on {self.device}...")
        self.agent = EnsembleAgent(
            time_series_dim=self.num_features,
            vision_channels=self.vision_channels,
            action_dim=self.action_dim,
            device=self.device
        )
        
        # Load weights based on mode
        model_path = None
        if self.mode == 'swing':
            model_path = str(SWING_MODEL_PATH)
        elif self.mode == 'day':
            model_path = str(SCALPER_MODEL_PATH)
        elif self.mode == 'crypto':
            model_path = str(SHARED_MODEL_PATH)
            
        loaded = False
        if model_path:
            prefix = model_path
            if "_balanced.pth" in model_path:
                prefix = model_path.replace("_balanced.pth", "")
            elif "_aggressive.pth" in model_path:
                prefix = model_path.replace("_aggressive.pth", "")
            elif "_conservative.pth" in model_path:
                prefix = model_path.replace("_conservative.pth", "")
            
            if os.path.exists(f"{prefix}_balanced.pth"):
                print(f"📂 Loading Configured Model: {prefix}...")
                self.agent.load(prefix)
                loaded = True
        
        if not loaded:
            # Fallback to auto-discovery
            import glob
            checkpoints = glob.glob("models/ensemble_ep*_balanced.pth")
            if checkpoints:
                # Find latest
                ep_nums = []
                for cp in checkpoints:
                    try:
                        num = int(cp.split("ensemble_ep")[1].split("_")[0])
                        ep_nums.append(num)
                    except: pass
                
                if ep_nums:
                    latest_ep = max(ep_nums)
                    print(f"📂 Loading Ensemble Model (Episode {latest_ep})...")
                    self.agent.load(f"models/ensemble_ep{latest_ep}")
                else:
                    print("⚠️ No ensemble checkpoints found. Starting fresh.")
            else:
                print("⚠️ No pre-trained model found. Starting fresh.")

        # 4. Initialize Eyes & Ears (Data Pipeline)
        feed = getattr(self.config, 'DATA_FEED', 'sip')
        self.pipeline = MultiModalDataPipeline(window_size=self.window_size, creds=self.creds, feed=feed)
        
        # 5. WebSocket Stream for Day Trader (real-time 1min -> 5min aggregation)
        self.ws_stream = None
        if self.mode == 'day':
            print("📡 Initializing WebSocket stream for real-time data...")
            self.ws_stream = AlpacaWebSocketStream(self.symbols)
            # Seed with historical data before starting stream
            self._seed_websocket_history()
            self.ws_stream.start()
        
        # 6. PDT Protection for Swing Mode (accounts under $25k)
        self.pdt_protection = (self.mode == 'swing')  # Only swing bot needs PDT protection
        self.buy_times: Dict[str, datetime] = {}  # Track when positions were opened
        self.min_hold_hours = 24  # Minimum hold time to avoid day trade (hold overnight)

        # 7. Supertrend Strategy (Crypto Only)
        self.supertrend = None
        if self.mode == 'crypto':
            # Default Crypto Settings: ATR 10, Multiplier 3.0
            print("🚀 Initializing Supertrend Strategy (Hybrid Logic)...")
            self.supertrend = SupertrendStrategy(atr_period=10, multiplier=3.0)
            
        # 8. Local State Cache (To avoid API Rate Limits)
        self.cached_positions = []
        self.cached_cash = 0.0
        self.cached_buying_power = 0.0
        self.last_cache_update = datetime.now()
        
        print(f"✅ Bot Ready. Watching {len(self.symbols)} symbols.")
        if self.pdt_protection:
            print(f"🛡️ PDT Protection ENABLED - Min hold time: {self.min_hold_hours}h")

    def refresh_account_state(self):
        """Pre-fetch account and positions to cache for execution loop."""
        try:
            # 1. Get Positions
            self.cached_positions = self.api.list_positions()
            
            # 2. Get Cash & Buying Power
            account = self.api.get_account()
            self.cached_cash = float(account.cash)
            self.cached_buying_power = float(account.buying_power)
            
            # Print Summary
            print(f"\n💰 Equity: ${float(account.equity):.2f} | Cash: ${self.cached_cash:.2f} | BP: ${self.cached_buying_power:.2f}")
            if self.cached_positions:
                print(f"📦 Holding {len(self.cached_positions)} positions:")
                for p in self.cached_positions:
                    print(f"   • {p.symbol}: {p.qty}")
            else:
                print("📦 No open positions.")
                
        except Exception as e:
            print(f"⚠️ Error refreshing account state: {e}")

    
    def _seed_websocket_history(self):
        """Seed WebSocket aggregator with historical 5-min bars."""
        if not self.ws_stream:
            return
        print("📥 Seeding WebSocket with historical 5-min bars...")
        for symbol in self.symbols:
            try:
                df = self.pipeline.market_fetcher.get_bars(symbol, lookback_days=5, timeframe='5Min')
                if df is not None and not df.empty:
                    self.ws_stream.seed_history(symbol, df)
                    print(f"   ✅ {symbol}: {len(df)} historical bars loaded")
            except Exception as e:
                print(f"   ⚠️ {symbol}: Failed to seed history - {e}")

    def _load_watchlist(self):
        path = self.watchlist_path
        
        # If no path provided, try default locations based on mode
        if not path:
            if self.mode == 'crypto':
                path = os.path.join("config", "watchlists", "crypto_watchlist.txt")
            elif self.mode == 'day':
                path = os.path.join("config", "watchlists", "day_trade_list.txt")
            else:
                path = os.path.join("config", "watchlists", "my_portfolio.txt")

        # Fallback to root if not found
        if not os.path.exists(path):
            print(f"⚠️ Watchlist {path} not found. Checking root...")
            path = os.path.basename(path)
            
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    symbols = [line.strip() for line in f if line.strip()]
                print(f"📋 Loaded {len(symbols)} symbols from {path}")
                return symbols
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        
        print(f"⚠️ Watchlist not found or empty. Using default.")
        return ["AAPL", "TSLA", "NVDA", "AMD", "MSFT"]

    def _get_websocket_data(self, symbol: str):
        """
        Get data from WebSocket stream for Day Trader.
        Returns same format as pipeline.fetch_and_process().
        """
        import numpy as np
        from src.core.indicators import add_technical_indicators
        from src.core.state import normalize_window_state
        
        if not self.ws_stream:
            return None
        
        # Get 5-min bars from WebSocket aggregator
        df = self.ws_stream.get_bars(symbol)
        
        if df is None or len(df) < self.window_size + 20:
            print(f"   ⚠️ WebSocket: Insufficient bars for {symbol} (Got {len(df) if df is not None else 0}, need {self.window_size + 20})")
            # Fallback to REST API
            return self.pipeline.fetch_and_process(symbol, timeframe='5Min')
        
        # Add indicators
        df = add_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < self.window_size:
            return self.pipeline.fetch_and_process(symbol, timeframe='5Min')
        
        # Normalize state
        current_step = len(df) - 1
        ts_state = normalize_window_state(df, current_step, self.window_size)
        current_price = df['Close'].iloc[-1]
        
        # Get news and tokenize (same as pipeline)
        headlines = self.pipeline.news_fetcher.get_headlines(symbol, limit=3)
        combined_text = " ".join(headlines)
        
        encoding = self.pipeline.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return ts_state, input_ids, attention_mask, current_price

    def execute_trade(self, symbol, action, current_price):
        """
        Action Map: 0=Hold, 1=Buy, 2=Sell
        """
        if action == 0:
            print(f"   ⏸️ {symbol}: HOLD")
            return

        try:
            # Use Cached Positions
            positions_dict = {p.symbol: p for p in self.cached_positions}
            has_position = symbol in positions_dict
            qty_held = float(positions_dict[symbol].qty) if has_position else 0.0
            
            # Filter positions by asset class (Crypto vs Equity)
            target_asset_class = 'crypto' if self.mode == 'crypto' else 'us_equity'
            my_positions = [p for p in self.cached_positions if p.asset_class == target_asset_class]
            
            # Determine TIF
            tif = 'gtc' if self.mode == 'crypto' else 'day'
            
            if action == 1: # BUY
                if has_position:
                    print(f"   ⚠️ {symbol}: Already holding {qty_held}. Skipping BUY.")
                    return
                
                # Track buy time for PDT protection
                if self.pdt_protection:
                    self.buy_times[symbol] = datetime.now()
                
                # 1. Check Max Positions
                if len(my_positions) >= self.config.MAX_POSITIONS:
                    print(f"   ⚠️ Max positions reached ({len(my_positions)}/{self.config.MAX_POSITIONS}) for {target_asset_class}. Skipping BUY.")
                    return

                # 2. Dynamic Position Sizing (Use Buying Power for Margin Support)
                # Was: cash = self.cached_cash (Too conservative for margin accounts)
                cash = self.cached_buying_power
                
                # Slots remaining
                slots_left = self.config.MAX_POSITIONS - len(my_positions)
                
                # Calculate allocation
                if slots_left > 0:
                    target_val = cash / slots_left
                else:
                    target_val = 0 # Should be caught by max positions check
                
                # Cap at 95% of cash
                target_val = min(target_val, cash * 0.95)

                if target_val < 10.0:
                    print(f"   ⚠️ Insufficient cash (${cash:.2f}) for {symbol} (Allocated: ${target_val:.2f}). Skipping.")
                    return

                # Calculate Quantity
                if current_price > 0:
                    qty = target_val / current_price
                else:
                    print(f"   ❌ Invalid price ${current_price}")
                    return

                # Rounding
                if self.mode == 'crypto':
                    qty = float(f"{qty:.4f}")
                else:
                    qty = int(qty)
                    if qty < 1:
                        print(f"   ⚠️ Calculated quantity 0 for {symbol} (${target_val:.2f}). Skipping.")
                        return
                
                print(f"   🚀 {symbol}: BUY {qty} @ ${current_price:.2f} (Est. ${qty*current_price:.2f})")
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force=tif
                )
                
                # Optimistic Update of Cache
                self.cached_cash -= (qty * current_price)
                # Mock a new position object (simplified)
                class MockPos:
                    def __init__(self, s, q, ac):
                        self.symbol = s
                        self.qty = q
                        self.asset_class = ac
                self.cached_positions.append(MockPos(symbol, qty, target_asset_class))
                
            elif action == 2:  # SELL
                if not has_position:
                    print(f"   ⚠️ {symbol}: No position to SELL.")
                    return
                
                # PDT Protection
                if self.pdt_protection and symbol in self.buy_times:
                    buy_time = self.buy_times[symbol]
                    hold_duration = datetime.now() - buy_time
                    min_hold = timedelta(hours=self.min_hold_hours)
                    if hold_duration < min_hold:
                        remaining = min_hold - hold_duration
                        print(f"   🛡️ PDT Protection: {symbol} must be held {remaining.seconds//3600}h {(remaining.seconds%3600)//60}m more. Skipping SELL.")
                        return
                
                print(f"   🔻 {symbol}: SELL {qty_held} @ ${current_price:.2f}")
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty_held,
                    side='sell',
                    type='market',
                    time_in_force=tif
                )
                
                # Optimistic Update
                self.cached_cash += (qty_held * current_price)
                self.cached_positions = [p for p in self.cached_positions if p.symbol != symbol]
                
        except Exception as e:
            print(f"   ❌ Order Failed: {e}")


    def liquidate_all_positions(self):
        print(f"🚨 LIQUIDATING ALL POSITIONS ({self.mode.upper()} Mode) 🚨")
        try:
            positions = self.api.list_positions()
            target_asset_class = 'crypto' if self.mode == 'crypto' else 'us_equity'
            
            for pos in positions:
                if pos.asset_class == target_asset_class:
                    print(f"   📉 Closing {pos.symbol} ({pos.qty})...")
                    # Crypto doesn't support 'day' TIF, use 'gtc'
                    tif = 'gtc' if self.mode == 'crypto' else 'day'
                    self.api.submit_order(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side='sell',
                        type='market',
                        time_in_force=tif
                    )
            print(f"✅ All {target_asset_class} positions liquidated.")
        except Exception as e:
            print(f"❌ Error liquidating positions: {e}")

    def _calculate_atr_trailing_stop(self, df, atr_period=14, multiplier=3.0):
        """
        Calculates Chandelier Exit (Long)
        Stop = Highest High (last N) - ATR(N) * Multiplier
        """
        try:
            # Handle case sensitivity for columns
            high = df['High'] if 'High' in df.columns else df['high']
            low = df['Low'] if 'Low' in df.columns else df['low']
            close = df['Close'] if 'Close' in df.columns else df['close']
            
            # Calculate TR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = tr.rolling(window=atr_period).mean()
            
            # Calculate Highest High over the period
            highest_high = high.rolling(window=atr_period).max()
            
            # Calculate Stop
            stop_price = highest_high - (atr * multiplier)

            atr_value = atr.iloc[-1]
            return stop_price.iloc[-1], atr_value
        except Exception as e:
            print(f"Error calculating trailing stop: {e}. Columns: {df.columns}")
            return None, None

    def run_loop(self):
        # Determine Timeframe
        timeframe = '1Min'
        if self.mode == 'swing':
            timeframe = '1D'
        elif self.mode == 'day':
            timeframe = '5Min'

        print(f"\n🚀 Starting Trading Loop in {self.mode.upper()} mode ({timeframe})...")
        
        # Initial Cleanup for Day Trader (if restarted mid-day or morning)
        if self.mode == 'day' and self.config.LIQUIDATE_EOD:
             # Check if we have positions that shouldn't be there?
             # For now, we rely on the loop logic, but we could force a check here.
             pass

        while True:
            # Check Market Hours if not Crypto
            if self.mode != 'crypto':
                try:
                    clock = self.api.get_clock()
                    now = clock.timestamp
                    next_open = clock.next_open
                    next_close = clock.next_close
                    
                    # 1. Market Closed Logic
                    if not clock.is_open:
                        time_to_open = (next_open - now).total_seconds()
                        sleep_time = max(60, time_to_open - 300) # Wake up 5 mins before open
                        print(f"🌙 Market Closed. Sleeping for {sleep_time/3600:.2f} hours until 5 mins before open...")
                        time.sleep(sleep_time)
                        
                        # WAKE UP (Beginning of Day)
                        # If Day Trader, ensure we start flat
                        if self.mode == 'day':
                            print("☀️ Market Opening Soon! Checking for stale positions...")
                            self.liquidate_all_positions()
                            
                        continue
                    
                    # 2. Day Trader End-of-Day Logic
                    if self.mode == 'day':
                        time_to_close = (next_close - now).total_seconds()
                        
                        # If less than 10 mins to close (600 seconds)
                        if time_to_close <= 600:
                            print(f"⏰ Market closing in {time_to_close/60:.1f} mins. Stopping scans.")
                            self.liquidate_all_positions()
                            
                            # Sleep until next open (minus 5 mins)
                            time_to_open = (next_open - now).total_seconds()
                            sleep_time = max(60, time_to_open - 300)
                            print(f"💤 Sleeping for {sleep_time/3600:.2f} hours until next market open...")
                            time.sleep(sleep_time)
                            continue

                except Exception as e:
                    print(f"⚠️ Error checking clock: {e}")
                    time.sleep(60)
            
            # --- Refresh Account State (CACHE) ---
            self.refresh_account_state()

            print(f"\n⏰ Scan Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # --- RANKING LOGIC START ---
            scored_opportunities = [] 
            
            print(f"🔍 Scanning {len(self.symbols)} symbols for RANKING (Split Phase)...")
            
            # --- PHASE 1: PARALLEL DATA FETCHING (I/O BOUND) ---
            fetch_start = time.time()
            data_results = []
            
            def fetch_symbol_task(symbol):
                """Phase 1: Pure Data Fetching"""
                try:
                    # 1. Fetch Data
                    if self.mode == 'day' and self.ws_stream:
                        data = self._get_websocket_data(symbol)
                    else:
                        data = self.pipeline.fetch_and_process(symbol, timeframe=timeframe)
                    
                    if data is None:
                        return None
                        
                    ts_state, text_ids, text_mask, current_price = data
                    # Reshape TS State: (Flattened) -> (Window, Features)
                    ts_state = ts_state.reshape(self.window_size, self.num_features)
                    
                    return (symbol, ts_state, text_ids, text_mask, current_price)
                except Exception:
                    return None

            # 20 Workers for Swing, 5 for others
            workers = 20 if self.mode == 'swing' else 5 
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_symbol = {executor.submit(fetch_symbol_task, sym): sym for sym in self.symbols}
                
                completed_count = 0
                total = len(self.symbols)
                
                for future in as_completed(future_to_symbol):
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"   ⏳ Fetched {completed_count}/{total}...", end='\r')
                        
                    res = future.result()
                    if res:
                        data_results.append(res)
            
            fetch_duration = time.time() - fetch_start
            print(f"\n   ✅ Data Fetch Complete in {fetch_duration:.2f}s. (Found {len(data_results)} valid valid items)")

            # --- PHASE 2: INFERENCE (CPU BOUND) ---
            # Process sequentially (or batched) to respect GIL and avoid overhead
            print(f"   🧠 Running Inference & Filters...")
            
            for symbol, ts_state, text_ids, text_mask, current_price in data_results:
                try:
                    # Agent Act
                    action, confidence = self.agent.act(ts_state, text_ids, text_mask, eval_mode=True, return_q=True)
                    
                    # --- FILTERS ---
                    # --- SWING TRADER: NATURAL TRAILING STOP (Chandelier Exit) ---
                    if self.mode == 'swing':
                        try:
                            df_daily = self.pipeline.market_fetcher.get_bars(symbol, lookback_days=40, timeframe='1D')
                            if df_daily is not None and not df_daily.empty and len(df_daily) > 20:
                                stop_price, atr_value = self._calculate_atr_trailing_stop(
                                    df_daily, atr_period=14, multiplier=self.config.TRAILING_ATR_MULT
                                )
                                if stop_price is not None:
                                    if current_price < stop_price:
                                        action = 2 # Force Sell
                        except Exception: pass

                    # --- CRYPTO TRADER: SUPERTREND FILTER ---
                    if self.mode == 'crypto' and self.supertrend:
                        try:
                            df_bars = self.pipeline.market_fetcher.get_bars(symbol, lookback_days=2, timeframe='1Min')
                            if df_bars is not None and len(df_bars) > 20:
                                df_st = self.supertrend.calculate_supertrend(df_bars.copy())
                                last_trend = df_st['trend'].iloc[-1]
                                if action == 1 and not last_trend:
                                    action = 0 # Block Buy
                                elif (action == 0 or action == 2):
                                    pass 
                        except Exception: pass

                    # Store
                    scored_opportunities.append({
                        'symbol': symbol,
                        'action': action,
                        'confidence': float(confidence),
                        'price': current_price
                    })

                    # Verbose Print
                    act_str = ["HOLD", "BUY", "SELL"][action]
                    print(f"   👉 {symbol}: {act_str} (Conf: {confidence:.3f})")

                except Exception as e:
                    print(f"   ❌ Inference failed for {symbol}: {e}")

            # --- RANKING LOGIC END ---
            
            # Sort by Confidence (Descending)
            # We want high confidence BUYs first.
            # actions: 0=HOLD, 1=BUY, 2=SELL.
            # We separate them.
            
            buys = [x for x in scored_opportunities if x['action'] == 1]
            sells = [x for x in scored_opportunities if x['action'] == 2]
            
            # Sort BUYs by confidence (Higher Q-value = Better)
            buys.sort(key=lambda x: x['confidence'], reverse=True)
            
            if buys:
                print(f"\n📊 TOP RANKED BUY OPPORTUNITIES (Execution Optimization Active):")
                for i, item in enumerate(buys[:10]):
                     print(f"   {i+1}. {item['symbol']} (Conf: {item['confidence']:.3f})")
                print("-" * 30)
            
            # Execute SELLs first (Free up cash)
            for item in sells:
                print(f"📉 Executing SELL for {item['symbol']} (Conf: {item['confidence']:.2f})")
                self.execute_trade(item['symbol'], 2, item['price'])
                
            # Execute BUYs (Top N)
            for item in buys:
                print(f"📈 Executing BUY for {item['symbol']} (Conf: {item['confidence']:.2f})")
                self.execute_trade(item['symbol'], 1, item['price'])

            # Sleep
            sleep_time = 30 if self.mode == 'day' else 60
            print(f"💤 Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchlist", type=str, help="Path to watchlist file")
    parser.add_argument("--mode", type=str, default="swing", choices=["day", "swing", "crypto"], help="Trading mode")
    args = parser.parse_args()
    
    bot = MultiModalTrader(watchlist_path=args.watchlist, mode=args.mode)
    bot.run_loop()
