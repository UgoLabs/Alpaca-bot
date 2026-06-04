import torch
import numpy as np
import pandas as pd
import os
from transformers import DistilBertTokenizer
from src.data.news_fetcher import NewsFetcher
from src.data.market_fetcher import MarketDataFetcher
from src.core.indicators import add_technical_indicators, add_intraday_features
from src.core.feature_sets import DAY_FEATURE_COLS, SWING_FEATURE_COLS, get_feature_cols
from src.data.csv_utils import read_swing_csv
from src.core.state import normalize_window_state

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
        # Day live inference: 'swing' (11) or 'day' (15) — must match loaded checkpoint.
        self.day_feature_set = "day"

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
                # Try creating an empty one if we have live data? No, need history for window
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
                # Get the live candle (dict)
                candle = self.live_daily_candles[symbol]
                candle_date = pd.to_datetime(candle['date']).normalize()
                
                # Check if we already have this date in the CSV (avoid dupes)
                last_dt = df.index[-1].normalize()
                
                if candle_date > last_dt:
                    # Append new row
                    new_row = pd.DataFrame({
                        'Open': [candle['open']],
                        'High': [candle['high']],
                        'Low': [candle['low']],
                        'Close': [candle['close']],
                        'Volume': [candle['volume']]
                    }, index=[candle_date])
                    df = pd.concat([df, new_row])
                elif candle_date == last_dt:
                    # Update (overwrite) the last row with latest data
                    df.iloc[-1] = [candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']]
            except Exception as e:
                pass # Fail silently on injection to keep moving

        return df

    def fetch_and_process(self, symbol: str, timeframe: str = '1Min'):
        """
        Returns a tuple of tensors: (ts_state, text_ids, text_mask)
        or None if data is insufficient.
        """
        # Determine lookback based on timeframe
        lookback = 5
        if timeframe == '1D':
            # Use local CSV for 1D data to avoid rate limits
            df = self._load_local_csv(symbol)
            if df.empty:
                return None
        elif timeframe == '15Min':
            # Sparse intraday bars + sma_200 burn-in -> need ~15 trading days
            # to keep >= window_size bars after add_technical_indicators().
            lookback = 15
            # 1. Fetch Market Data
            df = self.market_fetcher.get_bars(symbol, lookback_days=lookback, timeframe=timeframe)
        elif timeframe == '5Min':
            lookback = 10
            # 1. Fetch Market Data
            df = self.market_fetcher.get_bars(symbol, lookback_days=lookback, timeframe=timeframe)
        else:
            df = self.market_fetcher.get_bars(symbol, lookback_days=lookback, timeframe=timeframe)
        
        if df.empty:
            # print(f"   ⚠️ Insufficient data for {symbol} (Empty DataFrame)")
            return None
            
        if len(df) < self.window_size + 20:
            # print(f"   ⚠️ Insufficient data for {symbol} (Got {len(df)} bars, need {self.window_size + 20})")
            return None
            
        # 2. Add Indicators
        df = add_technical_indicators(df)
        # Intraday timeframes (day mode) use the richer intraday feature set;
        # daily/swing keeps its original 11 features so the swing model still loads.
        is_intraday = timeframe in ('15Min', '5Min', '1Min')
        if is_intraday:
            df = add_intraday_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < self.window_size:
            return None
            
        # 3. Normalize Time-Series State (Aligned with Training/Backtest)
        if is_intraday:
            feature_cols = get_feature_cols(getattr(self, "day_feature_set", "day"))
        else:
            feature_cols = get_feature_cols("swing")
        
        # Filter valid features
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < len(feature_cols):
            return None
            
        # Get raw features for the AVAILABLE history (to compute stats)
        # We use a lookback window for simple Z-score normalization
        # Similar to backtest_swing.py which uses the whole file mean/std
        features_all = df[valid_cols].values
        
        # Calculate stats (Z-Score)
        mean = np.mean(features_all, axis=0)
        std = np.std(features_all, axis=0) + 1e-8
        norm_features_all = (features_all - mean) / std
        
        # Slice the LAST window_size elements for inference
        if len(norm_features_all) >= self.window_size:
            ts_state_np = norm_features_all[-self.window_size:]
        else:
            # Pad if somehow short (should be caught by len(df) check above)
            pad_len = self.window_size - len(norm_features_all)
            ts_state_np = np.pad(norm_features_all, ((pad_len, 0), (0, 0)), mode='edge')

        # Convert to Tensor (Shape: [Window_Size, Num_Features])
        # Note: We do NOT flatten it here because the Agent expects [Batch, Seq, Feat]
        # and usually auto-unsqueezes the batch dim.
        ts_state = torch.FloatTensor(ts_state_np)
        
        current_price = df['Close'].iloc[-1]
        
        # 4. Fetch News
        headlines = self.news_fetcher.get_headlines(symbol, limit=3)
        combined_text = " ".join(headlines)
        
        # 5. Tokenize Text
        # We need fixed length for batching
        encoding = self.tokenizer(
            combined_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64 # Keep it short for speed
        )
        
        input_ids = encoding['input_ids'].squeeze(0) # (Seq_Len)
        attention_mask = encoding['attention_mask'].squeeze(0) # (Seq_Len)
        
        return ts_state, input_ids, attention_mask, current_price

if __name__ == "__main__":
    pipeline = MultiModalDataPipeline()
    result = pipeline.fetch_and_process("AAPL")
    if result:
        ts, ids, mask = result
        print(f"Time-Series Shape: {ts.shape}")
        print(f"Text IDs Shape: {ids.shape}")
        print(f"Text Mask Shape: {mask.shape}")
    else:
        print("Failed to fetch data.")
