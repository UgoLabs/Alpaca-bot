import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
from src.core.indicators import add_technical_indicators
from src.data.tokenizer import FinancialTokenizer

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, window_size=60, max_samples=None):
        self.window_size = window_size
        self.tokenizer = FinancialTokenizer(max_len=32)
        
        # Load Data
        self.data_frames = self._load_data(data_dir, max_samples)
        self.indices = self._build_indices()
        
        # Placeholder for News (In a real scenario, this would be a DB or CSV lookup)
        self.dummy_headlines = [
            "Market hits all time high as inflation cools",
            "Tech stocks rally on AI optimism",
            "Fed signals potential rate cuts later this year",
            "Earnings report shows strong growth for sector",
            "Geopolitical tensions rise, causing market uncertainty",
            "Oil prices surge amidst supply concerns",
            "Crypto markets volatile after regulatory news",
            "Investors cautious ahead of jobs report",
            "Blue chip stocks dividend yield increases",
            "Market correction expected by analysts"
        ]

    def _load_data(self, data_dir, max_samples):
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        data = []
        print(f"Loading data from {len(files)} files...")
        
        for f in files[:10]: # Limit to 10 files for now to speed up dev
            try:
                df = pd.read_csv(f)
                # Normalize columns
                df.columns = [c.lower() for c in df.columns]
                rename_map = {
                    'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume',
                    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                }
                df = df.rename(columns=rename_map)
                
                # Add Indicators
                df = add_technical_indicators(df)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Normalize Features (Simple MinMax or Z-Score)
                # For simplicity here, we do a rolling normalization in the get_item or just global
                # Let's do simple global normalization per file for now
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + [c for c in df.columns if c not in ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Keep only numeric
                df_numeric = df[numeric_cols].astype(float)
                
                # Z-Score Normalization
                df_norm = (df_numeric - df_numeric.mean()) / (df_numeric.std() + 1e-8)
                
                if len(df_norm) > self.window_size:
                    data.append(df_norm.values)
                    
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        return data

    def _build_indices(self):
        indices = []
        for i, series in enumerate(self.data_frames):
            # We can start from window_size up to len(series)
            # Each index is (series_idx, end_row_idx)
            num_windows = len(series) - self.window_size
            for j in range(num_windows):
                indices.append((i, j + self.window_size))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        series_idx, end_row = self.indices[idx]
        start_row = end_row - self.window_size
        
        # 1. Time-Series Data
        ts_data = self.data_frames[series_idx][start_row:end_row]
        ts_tensor = torch.FloatTensor(ts_data)
        
        # 2. Text Data (Simulated)
        # In reality: fetch news from timestamp of end_row
        headline = self.dummy_headlines[idx % len(self.dummy_headlines)]
        input_ids, attention_mask = self.tokenizer.tokenize(headline)
        
        # Squeeze batch dim from tokenizer output
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        
        return ts_tensor, input_ids, attention_mask

def create_dataloader(data_dir, batch_size=32):
    dataset = MultiModalDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
