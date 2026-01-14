import pandas as pd
import requests
from io import StringIO
import time

def scrape_finviz_news(symbol):
    """
    Scrapes the news table from Finviz for a specific symbol.
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}"
    
    # Finviz blocks python-requests, so we need a browser user-agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Fetching {url}...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse tables
        # Finviz news is usually in a table with id="news-table"
        # pd.read_html returns a list of dataframes
        tables = pd.read_html(StringIO(response.text), attrs={"id": "news-table"})
        
        if not tables:
            print("No news table found.")
            return []
            
        news_df = tables[0]
        # Columns are typically: Date/Time, Headline
        # The structure is simple but sometimes headers are missing
        
        headlines = []
        for index, row in news_df.iterrows():
            # Finviz table has 2 columns: Timestamp, Headline (+ Source implied in text sometimes)
            # We just want the headline
            if len(row) >= 2:
                headline = row[1]
                # Sometimes source is in the link/row, purely text scraping for now
                headlines.append(headline)
                
        return headlines[:5] # Return top 5
        
    except Exception as e:
        print(f"Error scraping Finviz: {e}")
        return []

if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Testing Finviz Scraper for {ticker}...")
    news = scrape_finviz_news(ticker)
    for i, n in enumerate(news):
        print(f"{i+1}. {n}")
