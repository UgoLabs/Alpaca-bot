import os
import sys

# Ensure the project root (which contains api.py and main.py) is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import alpaca_trade_api as tradeapi
from .DataLoader import DataLoader  # Import the base class

from api import paper_api, live_api  # Import from api.py

class AlpacaDataLoader(DataLoader):
    def __init__(self, symbol, timeframe='1D', start_date=None, end_date=None, paper_trading=True):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.api = paper_api if paper_trading else live_api
        
    def fetch_data(self):
        bars = self.api.get_bars(
            self.symbol,
            self.timeframe,
            start=self.start_date,
            end=self.end_date
        ).df
        
        self.data = pd.DataFrame({
            'Open': bars['open'],
            'High': bars['high'],
            'Low': bars['low'],
            'Close': bars['close'],
            'Volume': bars['volume']
        })
        return self.data