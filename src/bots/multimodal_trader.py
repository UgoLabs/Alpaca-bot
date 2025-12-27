import time
import os
import argparse
import torch
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Optional, Dict

from config.settings import (
    DayTraderCreds, SwingTraderCreds, MoneyScraperCreds, CryptoTraderCreds,
    DayTraderConfig, SwingTraderConfig, MoneyScraperConfig, CryptoTraderConfig,
    ALPACA_BASE_URL, SWING_MODEL_PATH, SCALPER_MODEL_PATH, SHARED_MODEL_PATH
)
from src.agents.ensemble_agent import EnsembleAgent
from src.data.pipeline import MultiModalDataPipeline
from src.data.websocket_stream import AlpacaWebSocketStream
from src.strategies.supertrend import SupertrendStrategy


class MultiModalTrader:
    """
    Live Trading Bot using the Multi-Modal Agent (Transformer + Vision + Text).
    """
    def __init__(self, watchlist_path: Optional[str] = None, mode: str = "swing"):
        print(f"🤖 Initializing Multi-Modal Trader ({mode.upper()} Mode)...")
        
        self.mode = mode.lower()
        self.watchlist_path = watchlist_path
        
        # Initialize Supertrend for Crypto and Day Mode
        self.supertrend = None
        if self.mode in ['crypto', 'day']:
            print(f"   🦸 Enabling Supertrend Strategy for {self.mode.upper()}...")
            self.supertrend = SupertrendStrategy(atr_period=10, multiplier=3.0)
        
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
        self.pipeline = MultiModalDataPipeline(window_size=self.window_size)
        
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
        
        print(f"✅ Bot Ready. Watching {len(self.symbols)} symbols.")
        if self.pdt_protection:
            print(f"🛡️ PDT Protection ENABLED - Min hold time: {self.min_hold_hours}h")
    
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

    def get_positions(self):
        positions = self.api.list_positions()
        return {p.symbol: p for p in positions}

    def execute_trade(self, symbol, action, current_price):
        """
        Action Map: 0=Hold, 1=Buy, 2=Sell
        """
        if action == 0:
            print(f"   ⏸️ {symbol}: HOLD")
            return

        try:
            # Check existing position
            positions_dict = self.get_positions()
            has_position = symbol in positions_dict
            qty_held = float(positions_dict[symbol].qty) if has_position else 0.0
            
            # Filter positions by asset class to respect per-bot limits
            # Crypto Bot counts Crypto; Day/Swing Bots count Equities
            all_positions_list = self.api.list_positions()
            target_asset_class = 'crypto' if self.mode == 'crypto' else 'us_equity'
            my_positions = [p for p in all_positions_list if p.asset_class == target_asset_class]
            
            # Determine TIF
            tif = 'gtc' if self.mode == 'crypto' else 'day'
            
            if action == 1: # BUY
                if has_position:
                    print(f"   ⚠️ {symbol}: Already holding {qty_held}. Skipping BUY.")
                    return
                
                # Track buy time for PDT protection
                if self.pdt_protection:
                    self.buy_times[symbol] = datetime.now()
                
                # 1. Check Max Positions (Filtered by Asset Class)
                if len(my_positions) >= self.config.MAX_POSITIONS:
                    print(f"   ⚠️ Max positions reached ({len(my_positions)}/{self.config.MAX_POSITIONS}) for {target_asset_class}. Skipping BUY.")
                    return

                # 2. Dynamic Position Sizing
                account = self.api.get_account()
                cash = float(account.cash)
                
                # Slots remaining
                slots_left = self.config.MAX_POSITIONS - len(my_positions)
                
                # Calculate allocation
                if slots_left > 0:
                    target_val = cash / slots_left
                else:
                    target_val = 0 # Should be caught by max positions check
                
                # Cap at 95% of cash to leave room for fees/slippage
                target_val = min(target_val, cash * 0.95)

                if target_val < 10.0: # Alpaca min is roughly $1 or $10 depending on asset, keeping $10 safe
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
                    qty = int(qty) # Stocks usually whole shares
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
                
            elif action == 2:  # SELL
                if not has_position:
                    print(f"   ⚠️ {symbol}: No position to SELL.")
                    return
                
                # PDT Protection: Check minimum hold time
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
            
            # --- Account Summary ---
            try:
                account = self.api.get_account()
                positions = self.api.list_positions()
                print(f"\n💰 Equity: ${float(account.equity):.2f} | Cash: ${float(account.cash):.2f}")
                if positions:
                    print(f"📦 Holding {len(positions)} positions:")
                    for p in positions:
                        print(f"   • {p.symbol}: {p.qty}")
                else:
                    print("📦 No open positions.")
            except Exception as e:
                print(f"⚠️ Error fetching account info: {e}")

            print(f"\n⏰ Scan Time: {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in self.symbols:
                try:
                    print(f"🔍 Scanning {symbol}...")
                    
                    # 1. Fetch Data - Use WebSocket for Day Trader, REST for others
                    if self.mode == 'day' and self.ws_stream:
                        # Get real-time 5-min bars from WebSocket aggregator
                        data = self._get_websocket_data(symbol)
                    else:
                        # Use REST API pipeline for Swing/Crypto
                        data = self.pipeline.fetch_and_process(symbol, timeframe=timeframe)
                    
                    if data is None:
                        print(f"   ⚠️ Insufficient data for {symbol}")
                        continue
                        
                    ts_state, text_ids, text_mask, current_price = data
                    
                    # Reshape TS State: (Flattened) -> (Window, Features)
                    ts_state = ts_state.reshape(self.window_size, self.num_features)
                    
                    # 2. Ask Agent
                    action = self.agent.act(ts_state, text_ids, text_mask, eval_mode=True)
                    
                    # --- SUPER TREND STRATEGY (AI + Supertrend) ---
                    if (self.mode == 'crypto' or self.mode == 'day') and self.supertrend:
                        try:
                            # Fetch data for Supertrend
                            df_st_data = None
                            if self.mode == 'day' and self.ws_stream:
                                df_st_data = self.ws_stream.get_bars(symbol)
                            else:
                                df_st_data = self.pipeline.market_fetcher.get_bars(symbol, lookback_days=2, timeframe=timeframe)

                            if df_st_data is not None and not df_st_data.empty and len(df_st_data) > 20:
                                # Calculate Supertrend
                                df_st = self.supertrend.calculate_supertrend(df_st_data)
                                is_bullish = df_st['trend'].iloc[-1]
                                st_val = df_st['supertrend'].iloc[-1]
                                
                                print(f"   🦸 Supertrend: {'BULLISH' if is_bullish else 'BEARISH'} (${st_val:.2f})")
                                
                                # Logic: Supertrend drives the trade. AI acts as confirmation or is overridden.
                                # User Request: "Buy when bullish, Sell when bearish"
                                
                                if is_bullish:
                                    # Supertrend says UP
                                    if action == 1: # AI BUY
                                        print(f"   ✅ Supertrend & AI agree: BULLISH. Executing BUY.")
                                    elif action == 0: # AI HOLD
                                        print(f"   🚀 Supertrend is BULLISH (AI Holding). Forcing BUY.")
                                        action = 1
                                    elif action == 2: # AI SELL
                                        print(f"   🛡️ Supertrend is BULLISH. Overriding AI Sell -> HOLD.")
                                        action = 0
                                else:
                                    # Supertrend says DOWN
                                    if action == 1: # AI BUY
                                        print(f"   🛑 Supertrend is BEARISH. Blocking AI Buy -> HOLD.")
                                        action = 0
                                    elif action == 2: # AI SELL
                                        print(f"   ✅ Supertrend & AI agree: BEARISH. Executing SELL.")
                                    elif action == 0: # AI HOLD
                                        # If we hold it, sell it.
                                        positions = self.get_positions()
                                        if symbol in positions:
                                            print(f"   📉 Supertrend is BEARISH. Forcing SELL.")
                                            action = 2
                                
                        except Exception as e:
                            print(f"   ⚠️ Supertrend Error: {e}")

                    # 3. Execute
                    action_str = ["HOLD", "BUY", "SELL"][action]
                    print(f"   🧠 Model Decision: {action_str}")
                    
                    self.execute_trade(symbol, action, current_price)
                    
                except Exception as e:
                    print(f"   ❌ Error processing {symbol}: {e}")
            
            # Day trader scans more frequently (every 30 sec), others every 60 sec
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
