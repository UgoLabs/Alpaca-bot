import os
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
from config.settings import ALPACA_BASE_URL, SwingTraderCreds
import yfinance as yf
import time
import random

class MarketDataFetcher:
    """
    Fetches market data from Alpaca.
    """
    def __init__(self, creds=None, feed='sip'):
        if creds is None:
            creds = SwingTraderCreds
            
        self.feed = feed
        self.api = tradeapi.REST(
            creds.API_KEY,
            creds.API_SECRET,
            ALPACA_BASE_URL,
            api_version='v2'
        )

    def get_bars(self, symbol: str, lookback_days: int = 5, timeframe: str = '1Min') -> pd.DataFrame:
        """
        Fetch bars for the last N days.
        timeframe: '1Min', '5Min', '1D'
        """
        end_dt = datetime.now()
        
        # If using SIP feed on a free plan, we must request data older than 15 mins.
        # We apply a 16-minute buffer to be safe.
        if self.feed == 'sip':
            end_dt = end_dt - timedelta(minutes=16)
        
        # Adjust lookback based on timeframe to ensure enough data
        if timeframe == '1D':
            lookback_days = max(lookback_days, 100) # Need more days for daily bars
        elif timeframe == '15Min':
            # 15Min bars are sparse (~26/day). add_technical_indicators burns
            # ~200 rows (sma_200) on dropna, so we need a healthy buffer to keep
            # >= window_size bars after indicators. ~15 trading days is safe and
            # still well under the 1Min pagination limit below.
            lookback_days = max(lookback_days, 15)
        elif timeframe == '5Min':
            lookback_days = max(lookback_days, 5)
        
        # Force more history for Crypto to ensure we fill the window
        if '/' in symbol:
            lookback_days = max(lookback_days, 7)
            
        start_dt = end_dt - timedelta(days=lookback_days)
        
        # Format as RFC3339 with Z (UTC) - assuming container is UTC or we want to treat it as such
        # We strip microseconds to be safe and ensure 'Z' is present if needed, 
        # but alpaca-trade-api might handle ISO strings better if they are clean.
        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Map string to Alpaca TimeFrame
        if timeframe == '1Min':
            tf = TimeFrame.Minute
        elif timeframe in ['5Min', '15Min']:
            # We will fetch 1Min and aggregate locally
            tf = TimeFrame.Minute
        elif timeframe == '1D':
            tf = TimeFrame.Day
            
            # Use YFinance for 1D bars for Stocks (Not Crypto) if it's not a crypto pair
            if '/' not in symbol:
                return self.get_history_yfinance(symbol, period="2y", interval="1d")
        else:
            tf = TimeFrame.Minute

        try:
            # Check for Crypto
            if '/' in symbol:
                bars = self.api.get_crypto_bars(
                    symbol,
                    tf,
                    start=start_str,
                    end=end_str,
                    limit=10000
                ).df
            else:
                bars = self.api.get_bars(
                    symbol,
                    tf,
                    start=start_str,
                    end=end_str,
                    adjustment='raw',
                    feed=self.feed,
                    # Headroom for ~15 days of 1Min bars (incl. extended hours)
                    # so the SDK paginates fully instead of truncating to the
                    # OLDEST 10k bars (which would leave us trading stale data).
                    limit=20000
                ).df
            
            if bars.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize columns
            # Alpaca returns: open, high, low, close, volume, trade_count, vwap
            # We need: Open, High, Low, Close, Volume
            df = bars.reset_index()
            
            # Rename columns to Capitalized
            col_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'timestamp': 'Date' # If reset_index gives timestamp
            }
            df = df.rename(columns=col_map)
            
            # Handle Index
            if 'Date' not in df.columns and 'timestamp' in df.columns:
                 df = df.rename(columns={'timestamp': 'Date'})
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Ensure we have data after renaming
            if df.empty:
                return pd.DataFrame()
            
            # Fast Aggregation for 5Min / 15Min
            if timeframe in ['5Min', '15Min']:
                # Resample 1Min -> 5Min/15Min
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }
                # Use pandas resample string based on timeframe
                resample_str = '5min' if timeframe == '5Min' else '15min'
                # Resample and drop incomplete bins
                df = df.resample(resample_str).agg(agg_dict).dropna()

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_history_yfinance(self, symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data using yfinance.
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        # Fix symbol format for Yahoo Finance (e.g., BRK.B -> BRK-B)
        if "." in symbol:
            symbol = symbol.replace(".", "-")

        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Optimized for speed (was 1.0s, now 0.1s)
                # We have 4 workers, so effective rate is ~30-40 req/sec max
                time.sleep(0.1 + random.uniform(0.01, 0.05))
                
                ticker = yf.Ticker(symbol)
                # Add timeout to prevent hanging forever
                df = ticker.history(period=period, interval=interval, timeout=10)
                
                if df.empty:
                    # Retry once if empty, sometimes yfinance flakes
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return pd.DataFrame()

                # Filter columns and ensure no timezone issues
                # Yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # Ensure index is datetime
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                    
                return df
                
            except Exception as e:
                # If rate limited, sleep and retry
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    sleep_time = 2 + (attempt * 2)
                    print(f"⏳ Rate limited on {symbol}, waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                elif "database is locked" in str(e).lower():
                    time.sleep(random.uniform(0.1, 1.5))
                else:
                    # e.g., No data found
                    return pd.DataFrame()
        
        return pd.DataFrame()

if __name__ == "__main__":
    mdf = MarketDataFetcher()
    # Test Alpaca
    # df = mdf.get_bars("AAPL", lookback_days=1)
    # print(df.head())
    
    # Test yfinance
    df_yf = mdf.get_history_yfinance("AAPL", period="1mo", interval="1d")
    print(df_yf.head())
