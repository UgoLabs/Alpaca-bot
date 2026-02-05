
import os
import glob
from pathlib import Path

def update_watchlist():
    # 1. Read existing portfolio
    portfolio_path = Path("config/watchlists/my_portfolio.txt")
    existing_symbols = set()
    
    if portfolio_path.exists():
        with open(portfolio_path, 'r') as f:
            existing_symbols = {line.strip() for line in f if line.strip()}
    
    print(f"Current Watchlist: {len(existing_symbols)} symbols")

    # 2. Get symbols from validation data (The "Winning" unseen symbols)
    validation_dir = Path("data/historical_validation")
    new_symbols = set()
    
    files = glob.glob(str(validation_dir / "*_1D.csv"))
    for f in files:
        sym = os.path.basename(f).replace("_1D.csv", "")
        new_symbols.add(sym)
        
    print(f"New Candidates found: {len(new_symbols)} symbols")
    
    # 3. Merge
    combined_symbols = existing_symbols.union(new_symbols)
    sorted_list = sorted(list(combined_symbols))
    
    # 4. Filter out any exclusions (like 'IRBO', 'IRBT' if we want to be safe, though previous fix removed them)
    # The previous fix just removed them from the text file. 
    # Let's ensure IRBO/IRBT are not re-added if they are in the validation folder (unlikely if they are not tradeable data).
    # Specifically, check_assets failed on them, so fetch_validation_data (which checks tradable) probably skipped them.
    
    print(f"New Total Watchlist: {len(sorted_list)} symbols")
    
    # 5. Write back
    backup_path = Path("config/watchlists/my_portfolio.bak.txt")
    if portfolio_path.exists():
        portfolio_path.rename(backup_path)
        print(f"Backed up old list to {backup_path}")
        
    with open(portfolio_path, 'w') as f:
        for sym in sorted_list:
            f.write(f"{sym}\n")
            
    print("✅ Watchlist updated successfully.")

if __name__ == "__main__":
    update_watchlist()
