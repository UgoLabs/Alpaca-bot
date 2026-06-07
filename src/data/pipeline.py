import torch
import numpy as np
import pandas as pd
import os
from transformers import DistilBertTokenizer
from src.data.news_fetcher import NewsFetcher
from src.data.market_fetcher import MarketDataFetcher
from src.core.indicators import add_technical_indicators
from src.core.feature_sets import get_feature_cols
from src.data.csv_utils import read_swing_csv

class MultiModalDataPipeline:
    """
    Orchestrates data fetching and preprocessing for the Multi-Modal Agent.
    """
    def __init__(self, window_size=60, creds=None, feed='sip'):
        self.window_size = window_size
        self.news_fetcher = NewsFetcher()
        self.market_fetcher = MarketDataFetcher(creds=creds, feed=feed)
        
        # Cache for local CSV data (swing trading)
        self._csv_cache = {}
        self._csv_dir = "data/historical_swing"
        
        # Real-time injection cache (populated by trader loop)
        self.live_daily_candles = {}

        # Initialize Tokenizer (cached)
        print("Loading Tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def _load_local_csv(self, symbol: str) -> pd.DataFrame:
        """Load historical data from local CSV (fast, no rate limits)."""
        if symbol in self._csv_cache:
            df = self._csv_cache[symbol].copy()
        else:
            csv_path = os.path.join(self._csv_dir, f"{symbol}_1D.csv")
            if not os.path.exists(csv_path):
                df = pd.DataFrame()
            else:
                try:
                    df = read_swing_csv(csv_path)
                    if not df.empty:
                        self._csv_cache[symbol] = df
                        df = df.copy()
                except Exception:
                    df = pd.DataFrame()

        # --- LIVE DATA INJECTION ---
        if not df.empty and symbol in self.live_daily_candles:
            try:
                candle = self.live_daily_candles[symbol]
                candle_date = pd.to_datetime(candle['date']).normalize()
                last_dt = df.index[-1].normalize()
                
                if candle_date > last_dt:
                    new_row = pd.DataFrame({
                        'Open': [candle['open']],
                        'High': [candle['high']],
                        'Low': [candle['low']],
                        'Close': [candle['close']],
                        'Volume': [candle['volume']]
                    }, index=[candle_date])
                    df = pd.concat([df, new_row])
                elif candle_date == last_dt:
                    df.iloc[-1] = [candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']]
            except Exception:
                pass

        return df

    def fetch_and_process(self, symbol: str, timeframe: str = '1D'):
        """
        Returns a tuple of tensors: (ts_state, text_ids, text_mask)
        or None if data is insufficient.
        """
        if timeframe != '1D':
            timeframe = '1D'

        df = self._load_local_csv(symbol)
        if df.empty:
            return None
        
        if len(df) < self.window_size + 20:
            return None
            
        df = add_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < self.window_size:
            return None
            
        feature_cols = get_feature_cols("swing")
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < len(feature_cols):
            return None
            
        features_all = df[valid_cols].values
        mean = np.mean(features_all, axis=0)
        std = np.std(features_all, axis=0) + 1e-8
        norm_features_all = (features_all - mean) / std
        
        if len(norm_features_all) >= self.window_size:
            ts_state_np = norm_features_all[-self.window_size:]
        else:
            pad_len = self.window_size - len(norm_features_all)
            ts_state_np = np.pad(norm_features_all, ((pad_len, 0), (0, 0)), mode='edge')

        ts_state = torch.FloatTensor(ts_state_np)
        current_price = df['Close'].iloc[-1]

        headlines = self.news_fetcher.get_headlines(symbol, limit=3)
        combined_text = " ".join(headlines)
        encoding = self.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64,
        )
        text_ids = encoding['input_ids'].squeeze(0)
        text_mask = encoding['attention_mask'].squeeze(0)

        return ts_state, text_ids, text_mask, current_price
