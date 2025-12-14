"""
Swing Trader Bot
Multi-day position trader using ATR-based risk management
"""
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config.settings import (
    SwingTraderCreds, SwingTraderConfig,
    ALPACA_BASE_URL, RISK_PER_TRADE, SWING_MODEL_PATH, SHARED_MODEL_PATH
)
from src.bots.base_bot import BaseBot
from src.core.indicators import add_technical_indicators
from src.core.state import normalize_state
from src.core.risk import calculate_position_size, calculate_atr_stops, check_risk_limits


class SwingTraderBot(BaseBot):
    """
    Swing trading bot for multi-day positions.
    - ATR-based stops and profit targets
    - Risk-based position sizing (matches training environment)
    - 15-minute scan interval
    """
    
    def __init__(self):
        # Use swing model if exists, otherwise fall back to shared
        import os
        model_path = str(SWING_MODEL_PATH) if os.path.exists(SWING_MODEL_PATH) else str(SHARED_MODEL_PATH)
        
        super().__init__(
            api_key=SwingTraderCreds.API_KEY,
            api_secret=SwingTraderCreds.API_SECRET,
            model_path=model_path,
            watchlist_file=SwingTraderConfig.WATCHLIST
        )
        
        self.config = SwingTraderConfig
        
        print(f"🎯 Risk/Trade: {RISK_PER_TRADE*100}%")
        print(f"📊 ATR Stops: {self.config.STOP_ATR_MULT}x / Trail: {self.config.TRAILING_ATR_MULT}x")
    
    def get_data(self, symbol):
        """Get daily data for swing trading."""
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            
            bars = self.api.get_bars(
                symbol,
                '1Day',
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=252,
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
        """Check positions for ATR-based exit signals."""
        try:
            positions = self.api.list_positions()
            if not positions:
                return
            
            print(f"\n💼 Managing {len(positions)} positions...")
            
            for p in positions:
                try:
                    symbol = p.symbol
                    entry_price = float(p.avg_entry_price)
                    current_price = float(p.current_price)
                    pnl_pct = float(p.unrealized_plpc)
                    total_pnl = float(p.unrealized_pl)
                    
                    # Get current data for ATR
                    df = self.get_data(symbol)
                    if df is None:
                        continue
                    
                    atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                    stops = calculate_atr_stops(
                        entry_price, atr,
                        self.config.STOP_ATR_MULT,
                        self.config.TRAILING_ATR_MULT,
                        self.config.PROFIT_ATR_MULT
                    )
                    
                    should_exit = False
                    exit_reason = ""
                    
                    # Hard stop loss
                    if current_price <= stops['stop_loss']:
                        should_exit = True
                        exit_reason = "STOP"
                    
                    # Take profit
                    elif current_price >= stops['take_profit']:
                        should_exit = True
                        exit_reason = "PROFIT"
                    
                    # Trailing stop check
                    elif symbol in self.position_tracking:
                        tracking = self.position_tracking[symbol]
                        if current_price > tracking.get('peak_price', entry_price):
                            tracking['peak_price'] = current_price
                        
                        # If we've hit trailing trigger and then fallen back
                        if tracking['peak_price'] >= stops['trailing_trigger']:
                            trail_stop = tracking['peak_price'] - (atr * 2.0)
                            if current_price <= trail_stop:
                                should_exit = True
                                exit_reason = "TRAIL"
                    
                    if should_exit:
                        self.api.close_position(symbol)
                        emoji = "✅" if exit_reason == "PROFIT" else "🛑"
                        print(f"   {emoji} {symbol:6s} {exit_reason}: {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                        if symbol in self.position_states:
                            del self.position_states[symbol]
                        if symbol in self.position_tracking:
                            del self.position_tracking[symbol]
                    else:
                        print(f"   💎 {symbol:6s} {pnl_pct*100:+.2f}% (${total_pnl:+.2f})")
                
                except Exception as e:
                    print(f"   ⚠️ {p.symbol}: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error checking exits: {e}")
    
    def scan_for_entries(self):
        """Scan for swing trade opportunities."""
        try:
            account = self.api.get_account()
            positions_map = self.get_positions_map()
            
            # CRITICAL FIX: Only scan for entries near market close (Swing Trading)
            # Prevent buying on incomplete daily bars which fluctuate wildly.
            now_et = datetime.now(self.eastern)
            market_open, market_close = self.get_market_schedule()
            
            # Default cutoff: 30 mins before close (usually 3:30 PM ET)
            cutoff_time = market_close - timedelta(minutes=60) 
            
            if now_et < cutoff_time:
                print(f"⏳ Daily candles incomplete. Walking forward exits only. Entries allowed after {cutoff_time.strftime('%H:%M')} ET.")
                return

            equity = float(account.equity)
            buying_power = float(account.buying_power)
            potential_buys = []
            
            print(f"\n🔍 Scanning {len(self.symbols)} symbols...")
            
            for symbol in self.symbols:
                if symbol in positions_map:
                    continue
                
                df = self.get_data(symbol)
                if df is None or len(df) < 50:
                    continue
                
                try:
                    # Get AI decision
                    state = normalize_state(df, len(df) - 1, 20)
                    
                    # Add portfolio features
                    current_price = float(df['Close'].iloc[-1])
                    portfolio_state = np.array([
                        equity / 100000,
                        buying_power / equity if equity > 0 else 0,
                        len(positions_map) / 100,
                        0,  # No existing position
                        0
                    ])
                    full_state = np.concatenate([state, portfolio_state])
                    
                    action = self.agent.act(full_state)
                    
                    if action == 1:  # BUY
                        # ATR-based position sizing
                        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                        stop_distance = atr * self.config.STOP_ATR_MULT
                        
                        qty = calculate_position_size(equity, current_price, stop_distance)
                        
                        if qty > 0:
                            # Get confidence from Q-values
                            import torch
                            state_tensor = torch.FloatTensor(full_state).unsqueeze(0)
                            with torch.no_grad():
                                q_values = self.agent.model(state_tensor.to(self.agent.device))
                                confidence = q_values[0][action].item()
                            
                            potential_buys.append({
                                'symbol': symbol,
                                'price': current_price,
                                'qty': qty,
                                'atr': atr,
                                'confidence': confidence,
                                'state': full_state
                            })
                
                except Exception as e:
                    continue
            
            # Execute top picks by confidence
            if potential_buys:
                potential_buys.sort(key=lambda x: x['confidence'], reverse=True)
                
                print(f"\n🎯 Processing {len(potential_buys)} buy signals:")
                for pick in potential_buys:
                    try:
                        self.api.submit_order(
                            symbol=pick['symbol'],
                            qty=pick['qty'],
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        self.position_states[pick['symbol']] = {
                            'state': pick['state'],
                            'entry_price': pick['price'],
                            'entry_time': datetime.now()
                        }
                        self.position_tracking[pick['symbol']] = {
                            'entry_price': pick['price'],
                            'entry_atr': pick['atr'],
                            'peak_price': pick['price']
                        }
                        
                        print(f"   🟢 {pick['symbol']:6s} BUY {pick['qty']} @ ${pick['price']:.2f} (conf: {pick['confidence']:.3f})")
                    
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "insufficient buying power" in err_msg:
                            print(f"   💰 Low buying power - stopping")
                            break
                        print(f"   ❌ {pick['symbol']:6s} FAILED: {str(e)[:40]}")
        
        except Exception as e:
            print(f"Error scanning: {e}")
    
    def run_once(self):
        """Run one trading cycle."""
        print(f"\n{'='*60}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Swing Trader Scan")
        print(f"{'='*60}")
        
        self.check_exits()
        self.scan_for_entries()
        
        # Online learning
        if len(self.replay_buffer) >= 64:
            loss = self.train_on_experiences()
            if loss > 0:
                print(f"\n🧠 Trained on {len(self.replay_buffer)} experiences (Loss: {loss:.4f})")
        
        print(f"\n{'='*60}")


def main():
    bot = SwingTraderBot()
    bot.run_loop(SwingTraderConfig.SCAN_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
