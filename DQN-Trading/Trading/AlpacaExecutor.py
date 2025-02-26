import logging
import time
import pandas as pd
import os
from datetime import datetime
from api import paper_api, live_api

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/alpaca_executor.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class AlpacaExecutor:
    def __init__(self, use_paper=True, use_live=False, live_ratio=0.01):
        self.use_paper = use_paper
        self.use_live = use_live
        self.live_ratio = live_ratio
        self.paper_api = paper_api
        self.live_api = live_api
        self.trade_log = []
        
        if use_paper:
            self._validate_api(self.paper_api, "paper")
        if use_live:
            self._validate_api(self.live_api, "live")
            
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _validate_api(self, api, account_type):
        try:
            account = api.get_account()
            logger.info(f"{account_type.capitalize()} Account: Cash: ${float(account.cash):.2f} | Portfolio Value: ${float(account.portfolio_value):.2f}")
        except Exception as e:
            logger.error(f"Error connecting to {account_type} API: {e}")
            raise
    
    def is_market_open(self):
        clock = self.paper_api.get_clock()
        return clock.is_open
    
    def get_next_market_open(self):
        clock = self.paper_api.get_clock()
        return clock.next_open
    
    def execute_trade(self, symbol, qty, side, wait_for_market=False, model_name="unknown", extended_hours=False):
        if not self.is_market_open():
            if wait_for_market:
                next_open = self.get_next_market_open()
                wait_time = (next_open - datetime.now()).total_seconds()
                if wait_time > 0:
                    logger.info(f"Market closed. Waiting {wait_time/60:.1f} minutes for market open.")
                    time.sleep(wait_time)
            else:
                logger.info("Market closed. Order not placed.")
                return False
        
        live_qty = max(1, int(qty * self.live_ratio)) if self.live_ratio > 0 else 0
        
        trade_info = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'paper_qty': qty if self.use_paper else 0,
            'live_qty': live_qty if self.use_live else 0,
            'model': model_name,
            'paper_executed': False,
            'live_executed': False,
            'paper_error': None,
            'live_error': None
        }
        
        # Execute on paper account
        if self.use_paper:
            try:
                self.paper_api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day',
                    extended_hours=extended_hours
                )
                trade_info['paper_executed'] = True
                logger.info(f"PAPER: {side.upper()} {qty} {symbol}")
            except Exception as e:
                trade_info['paper_error'] = str(e)
                logger.error(f"Paper trading error: {e}")
        
        # Execute on live account with risk controls
        if self.use_live:
            try:
                account = self.live_api.get_account()
                portfolio_value = float(account.portfolio_value)
                max_trade_value = portfolio_value * 0.2  # 20% risk per trade
                last_quote = self.live_api.get_latest_quote(symbol)
                price = (last_quote.bidprice + last_quote.askprice) / 2
                risk_adjusted_qty = int(max_trade_value / price)
                live_qty = min(live_qty, risk_adjusted_qty)
                
                if live_qty >= 1:
                    self.live_api.submit_order(
                        symbol=symbol,
                        qty=live_qty,
                        side=side,
                        type='market',
                        time_in_force='day'
                    )
                    trade_info['live_executed'] = True
                    trade_info['live_qty'] = live_qty
                    logger.info(f"LIVE: {side.upper()} {live_qty} {symbol}")
                else:
                    trade_info['live_error'] = "Quantity too small or exceeds risk limit"
                    logger.info("Live trade skipped: quantity too small or exceeds risk limit")
            except Exception as e:
                trade_info['live_error'] = str(e)
                logger.error(f"Live trading error: {e}")
        
        self.trade_log.append(trade_info)
        self._save_trade_log()
        return trade_info['paper_executed'] or trade_info['live_executed']
    
    def execute_limit_order(self, symbol, qty, side, limit_price, model_name="unknown"):
        try:
            self.paper_api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force='gtc',
                limit_price=limit_price
            )
            logger.info(f"LIMIT ORDER: {side.upper()} {qty} {symbol} at {limit_price}")
        except Exception as e:
            logger.error(f"Limit order error: {e}")
    
    def _save_trade_log(self):
        log_file = os.path.join(self.log_dir, f'trades_{datetime.now().strftime("%Y%m%d")}.csv')
        pd.DataFrame(self.trade_log).to_csv(log_file, index=False)
        
    def get_positions(self):
        results = {'paper': None, 'live': None}
        if self.use_paper:
            try:
                results['paper'] = self.paper_api.list_positions()
            except Exception as e:
                logger.error(f"Error getting paper positions: {e}")
        if self.use_live:
            try:
                results['live'] = self.live_api.list_positions()
            except Exception as e:
                logger.error(f"Error getting live positions: {e}")
        return results
