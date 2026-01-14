import pandas as pd
import os
import sys

# Ensure we can import from src if needed, though this script uses standard libs mostly
sys.path.append(os.getcwd())

def update_portfolio():
    # 1. Get current portfolio
    portfolio_path = os.path.join(os.getcwd(), 'config', 'watchlists', 'my_portfolio.txt')
    if os.path.exists(portfolio_path):
        with open(portfolio_path, 'r') as f:
            current_symbols = set(line.strip() for line in f if line.strip())
    else:
        current_symbols = set()
    
    print(f"Current portfolio has {len(current_symbols)} symbols")

    # 2. Get S&P 500 symbols
    try:
        print("Fetching S&P 500 list from Wikipedia...")
        # Use simple read_html with storage_options to set User-Agent
        dfs = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            storage_options={'User-Agent': 'Mozilla/5.0'}
        )
        sp500_df = dfs[0]
        # Clean symbols: Wikipedia uses '.' for BRK.B, Alpaca/files usually use '.' or '-'? 
        # Check existing data folder.
        # Usually Alpaca uses names like BRK.B, but filenames might be BRK-B?
        # Let's standardize on what is in the table for now, maybe replace '.' with '-' if that's the convention.
        # Checking existing files from workspace info: "AAPL_1Min.csv", "BRK.B" usually isn't shown in the limited list.
        # Let's just assume list format is standard.
        sp500_symbols = set(sp500_df['Symbol'].tolist())
        print(f"Fetched {len(sp500_symbols)} S&P 500 symbols")
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        # Fallback or exit
        return

    # 3. Merge (Union)
    new_portfolio = current_symbols.union(sp500_symbols)
    
    print(f"New portfolio has {len(new_portfolio)} symbols")
    print(f"Added {len(new_portfolio) - len(current_symbols)} new symbols")

    # 4. Save
    with open(portfolio_path, 'w') as f:
        for sym in sorted(new_portfolio):
            f.write(f"{sym}\n")
    
    print(f"Updated {portfolio_path}")

if __name__ == "__main__":
    update_portfolio()
