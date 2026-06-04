import yfinance as yf
import time
import requests
import pandas as pd
from io import StringIO
from typing import List, Optional

import json
import os
import random

import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class NewsFetcher:
    """
    Fetches financial news using yfinance.
    """
    def __init__(self, creds=None):
        self._cache = {} # {symbol: [headlines]}
        self._cache_timestamp = {} # {symbol: time_fetched}
        self.CACHE_DURATION = 3600 * 4 # 4 hour cache validity (Reduced stress)
        self._circuit_broken = False
        self._failure_count = 0
        self._max_failures = 20 # Tolerant
        self.cache_file = "data/news_cache.json"
        
        # Load persistent cache
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self._cache = data.get('news', {})
                    self._cache_timestamp = data.get('timestamps', {})
                print(f"   📋 Loaded {len(self._cache)} cached news entries from disk.")
            except Exception as e:
                print(f"   ⚠️ Failed to load news cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'news': self._cache,
                    'timestamps': self._cache_timestamp
                }, f)
        except Exception as e:
            # excessive printing can clog logs, ignore
            pass
        
    def prefetch_news(self, symbols: List[str]):
        """
        Prefetch news for all symbols to populate cache.
        """
        # Skip prefetch to avoid Rate Limits on startup for large lists
        # But now with disk cache, we can check if we truly NEED to fetch details
        
        # Count how many are MISSING or EXPIRED
        needed = [s for s in symbols if s not in self._cache or (time.time() - self._cache_timestamp.get(s, 0) > self.CACHE_DURATION)]
        
        if len(needed) > 50:
             print(f"   ℹ️ Skipping News Prefetch for {len(needed)} symbols to avoid Rate Limits. Will fetch Just-In-Time.")
             return

        print(f"   🗞️ Prefetching news for {len(needed)} symbols...")
        count = 0
        total = len(needed)
        
        for symbol in needed:
            if self._circuit_broken:
                break
                
            count += 1
            if count % 20 == 0:
                print(f"      fetched {count}/{total}...", end='\r')

            # Populate cache (this also saves to disk now)
            self.get_headlines(symbol, limit=3)
            
            # Small sleep to be nice to API
            time.sleep(0.5) 
        
        print(f"      ✅ News pre-fetch complete.")

    def get_headlines(self, symbol: str, limit: int = 5) -> List[str]:
        """
        Fetch latest news headlines for a symbol using yfinance.
        """
        # 0. Circuit Breaker
        if self._circuit_broken:
             # Even if broken, return CACHED data if valid
             if symbol in self._cache:
                 return self._cache[symbol][:limit]
             return ["Market analysis pending."]

        # 1. Check Cache
        if symbol in self._cache and symbol in self._cache_timestamp:
            if time.time() - self._cache_timestamp[symbol] < self.CACHE_DURATION:
                return self._cache[symbol][:limit]

        # 2. Fetch from yfinance
        try:
            # Jitter to prevent concurrent hits from parallel threads
            time.sleep(random.uniform(0.5, 1.5))
            
            # Suppress yfinance internal prints/errors using a simple check
            # We can't easily suppress C-level/Library prints without lower level hacks
            # But we can try/except the call.
            
            # Wrapper to fetch news with a strict timeout to avoid hanging
            # Replaces: news = ticker.news (which can hang)
            news = []
            try:
                # Direct API hit with timeout controls
                # This mimics what yfinance does but gives us control over the socket
                url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
                # Add a User-Agent to avoid immediate blocking
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                
                resp = requests.get(url, headers=headers, timeout=5) # 5 Second Strict Timeout
                if resp.status_code == 200:
                    data = resp.json()
                    news = data.get('news', [])
                elif resp.status_code == 429:
                    # Rate Limited
                    self._failure_count += 2 # Penalize heavier
                    raise Exception("Rate Limited 429")
            except Exception as req_e:
                # If direct fetch fails, we just assume no news rather than failing
                # print(f"Debug: News fetch error {symbol}: {req_e}")
                pass

            headlines = []
            if news:
                # Reset failure count on success
                self._failure_count = 0
                
                for item in news:
                    # Extract title - check multiple possible locations
                    title = "No Title"
                    if 'title' in item:
                        title = item['title']
                    elif 'content' in item and 'title' in item['content']:
                        title = item['content']['title']
                    
                    if title != "No Title":
                        headlines.append(title)
            else:
                 # If news call failed or returned empty
                 if self._failure_count > self._max_failures:
                     if not self._circuit_broken:
                        self._circuit_broken = True
                        print("\n   ⚠️ News Fetch Circuit Breaker Triggered (Too many failures). Disabling news for this session.")

            if not headlines:
                 # Fallback if no news found
                 headlines = ["No major news."]

            # Update Cache
            self._cache[symbol] = headlines
            self._cache_timestamp[symbol] = time.time()
            
            # Persist to disk regularly (on every 5th update or similar is better, but every write ensures safety)
            # Depending on I/O, maybe every write is okay for small file.
            self._save_cache()
            
            return headlines[:limit]

        except Exception as e:
            self._failure_count += 1
            # print(f"⚠️ News fetch failed for {symbol}: {e}")
            return ["Market analysis pending."]


if __name__ == "__main__":
    # Quick test
    nf = NewsFetcher()
    print(nf.get_headlines("AAPL"))
