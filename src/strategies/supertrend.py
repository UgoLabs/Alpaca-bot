import pandas as pd
import numpy as np

class SupertrendStrategy:
    def __init__(self, atr_period=10, multiplier=3.0):
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Supertrend indicator.
        Expects DataFrame with 'High', 'Low', 'Close' columns (capitalized).
        """
        # Ensure columns are lowercase for calculation if needed, or map them
        # The provided code used lowercase, but our pipeline uses Capitalized.
        # Let's map them to be safe.
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate True Range (TR)
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift(1))
        tr2 = abs(low - close.shift(1))
        df['tr'] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)

        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # Calculate Basic Bands
        hl2 = (high + low) / 2
        df['basic_upper'] = hl2 + (self.multiplier * df['atr'])
        df['basic_lower'] = hl2 - (self.multiplier * df['atr'])

        # Initialize Final Bands and Trend
        df['final_upper'] = 0.0
        df['final_lower'] = 0.0
        df['supertrend'] = 0.0
        df['trend'] = False  # True = Bullish, False = Bearish

        # Iterative calculation for Supertrend logic
        # We iterate starting from the first valid ATR index
        for i in range(self.atr_period, len(df)):
            # Final Upper Band Logic
            if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1]) or \
               (close.iloc[i-1] > df['final_upper'].iloc[i-1]):
                df.at[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
            else:
                df.at[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]

            # Final Lower Band Logic
            if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1]) or \
               (close.iloc[i-1] < df['final_lower'].iloc[i-1]):
                df.at[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
            else:
                df.at[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]

            # Trend Direction Logic
            # prev_supertrend = df['supertrend'].iloc[i-1] # Not strictly needed for logic below
            
            if df['trend'].iloc[i-1]: # Previously Bullish
                if close.iloc[i] < df['final_lower'].iloc[i]:
                    df.at[df.index[i], 'trend'] = False # Flip to Bearish
                    df.at[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                else:
                    df.at[df.index[i], 'trend'] = True # Stay Bullish
                    df.at[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
            else: # Previously Bearish
                if close.iloc[i] > df['final_upper'].iloc[i]:
                    df.at[df.index[i], 'trend'] = True # Flip to Bullish
                    df.at[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                else:
                    df.at[df.index[i], 'trend'] = False # Stay Bearish
                    df.at[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]

        return df

    def check_signal(self, df: pd.DataFrame):
        """
        Returns 'buy', 'sell', or None based on the latest closed candle.
        """
        if len(df) < self.atr_period + 2:
            return None

        # Check the last completed candle (iloc[-1])
        # Assuming the dataframe includes the most recent completed bar
        current_trend = df['trend'].iloc[-1]
        previous_trend = df['trend'].iloc[-2]

        if current_trend and not previous_trend:
            return 'buy'
        elif not current_trend and previous_trend:
            return 'sell'
        
        return None
