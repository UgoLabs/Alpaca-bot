"""
Day Trader Bot
Intraday scalping with tight stops
"""
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from config.settings import (
    DayTraderCreds, DayTraderConfig,
    ALPACA_BASE_URL, SHARED_MODEL_PATH
)
from src.bots.base_bot import BaseBot
from src.core.indicators import add_technical_indicators
from src.core.state import normalize_state
from src.core.risk import calculate_position_size, check_risk_limits


class DayTraderBot(BaseBot):
    """
    Intraday scalping bot.
    - Very tight profit/stop targets (0.5% / 0.3%)
    - Max 5 positions
    - 30-second scan interval
    """
    
    def __init__(self):
        super().__init__(
            api_key=DayTraderCreds.API_KEY,
            api_secret=DayTraderCreds.API_SECRET,
            model_path=str(SHARED_MODEL_PATH),
            watchlist_file=DayTraderConfig.WATCHLIST
        )
        
        self.config = DayTraderConfig
        
        print(f"🎯 Targets: +{self.config.PROFIT_TARGET_PCT*100}% / -{self.config.STOP_LOSS_PCT*100}%")
        print(f"🔢 Max Positions: {self.config.MAX_POSITIONS}")
    
    def get_data(self, symbol):
        """Get intraday data for day trading."""
        try:
            end = datetime.now()
            start = end - timedelta(days=2)
            
            bars = self.api.get_bars(
                symbol,
                '5Min',
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
        """Check positions for tight scalp targets."""
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
                        self.api.close_position(symbol)
                        print(f"   ✅ {symbol:6s} PROFIT: {pnl_pct*100:+.2f}%")
                    
                    # Stop loss
                    elif pnl_pct <= -self.config.STOP_LOSS_PCT:
                        self.api.close_position(symbol)
                        print(f"   🛑 {symbol:6s} STOP: {pnl_pct*100:+.2f}%")
                    
                    else:
                        print(f"   💎 {symbol:6s} {pnl_pct*100:+.2f}%")
                
                except Exception as e:
                    print(f"   ⚠️ {p.symbol}: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error checking exits: {e}")
    
    def scan_for_entries(self):
        """Scan for intraday scalp opportunities."""
        try:
            account = self.api.get_account()
            positions_map = self.get_positions_map()
            
            risk_check = check_risk_limits(account, list(positions_map.values()),
                                           self.config.MAX_POSITIONS)
            if not risk_check['can_trade']:
                print(f"\n⚠️ {risk_check['reason']}")
                return
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
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
                    
                    current_price = float(df['Close'].iloc[-1])
                    portfolio_state = np.array([
                        equity / 100000,
                        buying_power / equity if equity > 0 else 0,
                        len(positions_map) / self.config.MAX_POSITIONS,
                        0, 0
                    ])
                    full_state = np.concatenate([state, portfolio_state])
                    
                    action = self.agent.act(full_state)
                    
                    if action == 1:  # BUY
                        stop_distance = current_price * self.config.STOP_LOSS_PCT
                        qty = calculate_position_size(equity, current_price, stop_distance)
                        
                        if qty > 0:
                            potential_buys.append({
                                'symbol': symbol,
                                'price': current_price,
                                'qty': qty
                            })
                
                except Exception:
                    continue
            
            # Execute top picks
            if potential_buys:
                slots = risk_check['available_slots']
                top_picks = potential_buys[:slots]
                
                print(f"\n🎯 Executing {len(top_picks)} buys:")
                for pick in top_picks:
                    try:
                        self.api.submit_order(
                            symbol=pick['symbol'],
                            qty=pick['qty'],
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        print(f"   🟢 {pick['symbol']:6s} BUY {pick['qty']} @ ${pick['price']:.2f}")
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "insufficient buying power" in err_msg:
                            print("   💰 Low buying power - stopping")
                            break
                        print(f"   ❌ {pick['symbol']:6s} FAILED: {str(e)[:30]}")
        
        except Exception as e:
            print(f"Error scanning: {e}")
    
    def run_once(self):
        """Run one trading cycle."""
        print(f"\n{'='*60}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Day Trader Scan")
        print(f"{'='*60}")
        
        self.check_exits()
        self.scan_for_entries()
        
        print(f"\n{'='*60}")


def main():
    bot = DayTraderBot()
    bot.run_loop(DayTraderConfig.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
