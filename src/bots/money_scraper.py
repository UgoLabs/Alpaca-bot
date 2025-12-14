"""
Money Scraper Bot
High-frequency scalper using WebSocket for real-time data
"""
import time
from datetime import datetime, timedelta
from threading import Thread
from collections import deque

import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

from config.settings import (
    MoneyScraperCreds, MoneyScraperConfig, 
    ALPACA_BASE_URL, RISK_PER_TRADE, SCALPER_MODEL_PATH, SHARED_MODEL_PATH
)
from src.bots.base_bot import BaseBot
from src.core.indicators import add_technical_indicators
from src.core.state import normalize_state
from src.core.risk import calculate_position_size, check_risk_limits


class MoneyScraperBot(BaseBot):
    """
    High-frequency scalping bot.
    - Uses WebSocket for real-time 1-minute bars
    - Percentage-based profit/stop targets
    - Max 8 positions
    """
    
    def __init__(self):
        # Use scalper model if exists, otherwise fall back to shared
        import os
        model_path = str(SCALPER_MODEL_PATH) if os.path.exists(SCALPER_MODEL_PATH) else str(SHARED_MODEL_PATH)
        
        super().__init__(
            api_key=MoneyScraperCreds.API_KEY,
            api_secret=MoneyScraperCreds.API_SECRET,
            model_path=model_path,
            watchlist_file=MoneyScraperConfig.WATCHLIST
        )
        
        self.config = MoneyScraperConfig
        
        # WebSocket streaming
        if self.config.USE_WEBSOCKET:
            self.stream = None
            self.bar_cache = {symbol: deque(maxlen=200) for symbol in self.symbols}
            
            # Initialize WebSocket immediately if market is open
            if self.is_market_open():
                print("🔌 Market is OPEN - Initializing WebSocket immediately...")
                self._setup_websocket()
        else:
            self.stream = None
            self.bar_cache = {}
            
        print(f"🎯 Targets: +{self.config.PROFIT_TARGET_PCT*100}% / -{self.config.STOP_LOSS_PCT*100}%")
        print(f"🔢 Max Positions: {self.config.MAX_POSITIONS}")

    def on_warmup(self):
        """Start WebSocket connection 5 mins before open."""
        if self.config.USE_WEBSOCKET and self.stream is None:
            print("🔌 initializing WebSocket for Market Open...")
            self._setup_websocket()
    
    def on_shutdown(self):
        """Stop WebSocket connection after close."""
        if self.stream:
            print("🔌 Closing WebSocket connection...")
            try:
                # Alpaca Stream doesn't have a clean 'close' method exposed easily on the object wrapper sometimes,
                # but we can try to unsubscribe or just let it die.
                # Actually tradeapi.Stream.stop() or run loop stop is cleaner.
                # New v2 stream runs in asyncio loop.
                # Assuming 'run' method was threaded.
                # We can just nullify it and let the daemon thread die with the process or ignore.
                # But to be clean:
                if hasattr(self.stream, 'stop'):
                    self.stream.stop()
                if hasattr(self.stream, '_stop_stream'): # specific to implementation
                     self.stream._stop_stream()
            except:
                pass
            self.stream = None
            self.stream_ready = False
            
    def _setup_websocket(self):
        """Initialize WebSocket connection for streaming bars."""
        self.stream = tradeapi.Stream(
            MoneyScraperCreds.API_KEY,
            MoneyScraperCreds.API_SECRET,
            base_url=ALPACA_BASE_URL,
            data_feed='sip'
        )
        self.stream_ready = False
        
        async def bar_handler(bar):
            symbol = bar.symbol
            if symbol in self.bar_cache:
                self.bar_cache[symbol].append({
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
                
                # Check if stream is ready
                if not self.stream_ready:
                    bars_filled = sum(1 for s in self.bar_cache if len(self.bar_cache[s]) >= 10)
                    if bars_filled >= len(self.symbols) * 0.1:
                        self.stream_ready = True
                        print(f"🟢 WebSocket ready ({bars_filled}/{len(self.symbols)} symbols)")
        
        # Subscribe and start in background
        self.stream.subscribe_bars(bar_handler, *self.symbols)
        Thread(target=self.stream.run, daemon=True).start()
        print("🔌 WebSocket streaming started (1Min bars → 5Min aggregation)")
    
    def _aggregate_to_5min(self, bars_1min):
        """
        Aggregate 1-minute bars into 5-minute bars.
        This ensures the AI model receives data matching its training timeframe.
        """
        if len(bars_1min) < 5:
            return None
        
        df = pd.DataFrame(list(bars_1min))
        df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to 5-minute bars
        df_5min = df.resample('5Min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(df_5min) < 20:  # Need enough bars for indicators
            return None
        
        return df_5min.reset_index()
    
    def get_data(self, symbol):
        """Get market data for a symbol (WebSocket cache or REST fallback)."""
        # Try WebSocket cache first (aggregate 1Min → 5Min)
        if self.stream and symbol in self.bar_cache and len(self.bar_cache[symbol]) >= 100:
            df_5min = self._aggregate_to_5min(self.bar_cache[symbol])
            if df_5min is not None:
                return add_technical_indicators(df_5min)
        
        # REST API fallback - use 5Min bars directly (matches training!)
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            bars = self.api.get_bars(
                symbol,
                '5Min',  # Changed from 1Min to match training
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=200,
                feed='sip'
            ).df
            
            if bars.empty:
                return None
            
            df = bars.reset_index()
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })
            return add_technical_indicators(df)
        except Exception as e:
            return None
    
    def check_exits(self):
        """Check positions for profit/stop targets."""
        try:
            positions = self.api.list_positions()
            if not positions:
                return
            
            print(f"\n💼 Managing {len(positions)} positions...")
            
            for p in positions:
                try:
                    symbol = p.symbol
                    pnl_pct = float(p.unrealized_plpc)
                    total_pnl = float(p.unrealized_pl)
                    
                    # Profit target
                    if pnl_pct >= self.config.PROFIT_TARGET_PCT:
                        try:
                            self.api.cancel_all_orders()
                        except:
                            pass
                        self.api.close_position(symbol)
                        print(f"   ✅ {symbol:6s} PROFIT: {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                        if symbol in self.position_states:
                            del self.position_states[symbol]
                    
                    # Stop loss
                    elif pnl_pct <= -self.config.STOP_LOSS_PCT:
                        try:
                            self.api.cancel_all_orders()
                        except:
                            pass
                        self.api.close_position(symbol)
                        print(f"   🛑 {symbol:6s} STOP: {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                        if symbol in self.position_states:
                            del self.position_states[symbol]
                    
                    else:
                        print(f"   💎 {symbol:6s} {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                
                except Exception as e:
                    print(f"   ⚠️ {p.symbol}: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error checking exits: {e}")
    
    def scan_for_entries(self):
        """Scan for new entry opportunities."""
        try:
            account = self.api.get_account()
            positions_map = self.get_positions_map()
            
            risk_check = check_risk_limits(account, list(positions_map.values()), 
                                           self.config.MAX_POSITIONS)
            if not risk_check['can_trade']:
                print(f"\n⚠️ {risk_check['reason']}")
                return
            
            buying_power = float(account.buying_power)
            equity = float(account.equity)
            potential_buys = []
            
            print(f"\n🔍 Scanning {len(self.symbols)} symbols...")
            
            for symbol in self.symbols:
                if symbol in positions_map:
                    continue
                
                df = self.get_data(symbol)
                if df is None or len(df) < 20:
                    continue
                
                try:
                    # Get AI decision
                    state = normalize_state(df, len(df) - 1, 20)
                    
                    # Add portfolio features
                    current_price = float(df['Close'].iloc[-1])
                    portfolio_state = np.array([
                        equity / 100000,
                        buying_power / equity if equity > 0 else 0,
                        len(positions_map) / self.config.MAX_POSITIONS,
                        0,  # No existing position
                        0
                    ])
                    full_state = np.concatenate([state, portfolio_state])
                    
                    action = self.agent.act(full_state)
                    
                    if action == 1:  # BUY
                        # Calculate position size
                        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                        stop_distance = current_price * self.config.STOP_LOSS_PCT
                        
                        qty = calculate_position_size(equity, current_price, stop_distance)
                        
                        if qty > 0:
                            potential_buys.append({
                                'symbol': symbol,
                                'price': current_price,
                                'qty': qty,
                                'state': full_state
                            })
                
                except Exception as e:
                    continue
            
            # Execute top picks
            if potential_buys:
                slots = risk_check['available_slots']
                top_picks = potential_buys[:slots]
                
                print(f"\n🎯 Executing {len(top_picks)} buys:")
                for pick in top_picks:
                    try:
                        # Calculate prices for Bracket Order
                        tp_price = round(pick['price'] * (1 + self.config.PROFIT_TARGET_PCT), 2)
                        sl_price = round(pick['price'] * (1 - self.config.STOP_LOSS_PCT), 2)
                        
                        self.api.submit_order(
                            symbol=pick['symbol'],
                            qty=pick['qty'],
                            side='buy',
                            type='market',
                            time_in_force='day',
                            order_class='bracket',
                            take_profit={'limit_price': tp_price},
                            stop_loss={'stop_price': sl_price}
                        )
                        self.position_states[pick['symbol']] = {
                            'state': pick['state'],
                            'entry_price': pick['price'],
                            'entry_time': datetime.now()
                        }
                        print(f"   🟢 {pick['symbol']:6s} BUY {pick['qty']} @ ${pick['price']:.2f} (OCO: ${tp_price:.2f} / ${sl_price:.2f})")
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "insufficient buying power" in err_msg:
                            print(f"   💰 Low buying power - stopping")
                            break
                        print(f"   ❌ {pick['symbol']:6s} FAILED: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error scanning: {e}")
    
    def run_once(self):
        """Run one trading cycle."""
        print(f"\n{'='*60}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Money Scraper Scan")
        print(f"{'='*60}")
        
        self.check_exits()
        self.scan_for_entries()
        
        # Online Learning: Train on accumulated experiences
        if self.config.ONLINE_LEARNING and len(self.replay_buffer) >= 32:
            loss = self.train_on_experiences(batch_size=32)
            if loss > 0:
                print(f"🧠 Online Learning: Loss={loss:.4f} (Buffer: {len(self.replay_buffer)})")
        
        print(f"\n{'='*60}")


def main():
    bot = MoneyScraperBot()
    bot.run_loop(MoneyScraperConfig.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
