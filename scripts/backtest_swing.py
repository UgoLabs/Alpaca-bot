import sys
import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Force UTF-8 for Windows
if sys.platform == 'win32':
    # type: ignore
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TrainingConfig
# Override Window Size for Swing Trading
TrainingConfig.WINDOW_SIZE = 60

from src.environments.vector_env import VectorizedTradingEnv  # noqa: E402
from src.agents.ensemble_agent import EnsembleAgent  # noqa: E402
from src.core.indicators import add_technical_indicators  # noqa: E402

import yfinance as yf


def _compute_chandelier_stop(df: pd.DataFrame, atr_period: int = 14, atr_mult: float = 3.0) -> np.ndarray:
    """Chandelier Exit (long): stop = rolling_high(N) - ATR(N) * mult."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    hh = high.rolling(window=atr_period).max()
    stop = hh - (atr * atr_mult)

    # Make early values safe (no premature stops)
    stop = stop.bfill().fillna(0.0)
    return stop.values


def _compute_atr(df: pd.DataFrame, atr_period: int = 14) -> np.ndarray:
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    atr = atr.bfill().fillna(0.0)
    return atr.values

def _collect_known_symbols() -> set:
    """Collect symbols that are likely part of training/seen data to avoid leakage."""
    known: set[str] = set()

    # Symbols already present in local datasets
    for data_dir, suffix in [("data/historical_swing", "_1D.csv"), ("data/historical", "_1Min.csv")]:
        pattern = os.path.join(data_dir, f"*{suffix}")
        for path in glob.glob(pattern):
            sym = os.path.basename(path).replace(suffix, "")
            if sym:
                known.add(sym.upper())

    # Symbols present in any watchlist files
    wl_dir = os.path.join("config", "watchlists")
    if os.path.isdir(wl_dir):
        for wl_path in glob.glob(os.path.join(wl_dir, "*.txt")):
            try:
                with open(wl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        sym = line.strip()
                        if sym and not sym.startswith("#"):
                            known.add(sym.upper())
            except Exception:
                continue

    return known


# A broad candidate list across sectors; we filter against known/training symbols.
CANDIDATE_TEST_SYMBOLS = [
    "AAL","AAP","ABNB","ACI","ADSK","AES","AFRM","AKAM","ALB","ALLY",
    "AMCR","AMG","AMR","ANSS","AOS","APA","APD","ASML","ATO","AVB",
    "AVTR","AXP","AZO","BBWI","BEP","BKNG","BMRN","BNTX","BURL","CAG",
    "CAR","CARR","CCK","CHWY","CLX","CMA","CME","CNP","COO","CPB",
    "CPT","CTRA","CVS","D","DAL","DBX","DG","DKNG","DOV","DPZ",
    "DRI","DXCM","EBAY","EIX","EL","ENPH","EPAM","ETSY","EXR","F",
    "FAST","FCX","FDX","FICO","FSLR","FTNT","GD","GDDY","GIS","GL",
    "GM","GNRC","GRMN","HBI","HPE","HUM","IBKR","INTU","IP","IRM",
    "JKHY","KHC","KIM","KMX","KR","LVS","LYFT","MGM","MMM","MPC",
    "MTCH","NEM","NET","NKE","NTRS","NUE","OKTA","OTIS","PFG","PINS",
    "POOL","PRU","QRVO","REG","RIVN","ROKU","RSG","SBUX","SMCI","SNAP",
    "SNOW","SPLK","SQ","STX","SWK","SYY","TAP","TDG","TER","TROW",
    "TSCO","TTWO","TWLO","UAL","UBER","VFC","VLO","WBA","WDC","WEC",
    "WELL","WMB","WY","XEL","XYL","YUM","ZBH"
]


def download_test_data(
    output_dir: str = "data/test_swing",
    test_count: int = 40,
    period: str = "10y",
    interval: str = "1d",
    seed: int = 1337,
):
    """Downloads data for unseen symbols to test generalization."""
    os.makedirs(output_dir, exist_ok=True)

    known = _collect_known_symbols()
    rng = np.random.default_rng(seed)

    # Filter candidates against known symbols
    candidates = [s for s in CANDIDATE_TEST_SYMBOLS if s.upper() not in known]
    if not candidates:
        print("❌ No candidate test symbols left after filtering against known/training symbols.")
        return []

    # Sample without replacement
    pick_n = min(int(test_count), len(candidates))
    test_symbols = rng.choice(np.array(candidates, dtype=object), size=pick_n, replace=False).tolist()

    print(f"📥 Downloading Test Data for Generalization Check ({len(test_symbols)} symbols)")
    print(f"   Symbols: {test_symbols}")

    for symbol in tqdm(test_symbols, desc="Fetching Test Data"):
        try:
            path = os.path.join(output_dir, f"{symbol}_1D.csv")
            if os.path.exists(path):
                continue

            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df is None or df.empty:
                continue

            # Flatten MultiIndex if present (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(path)
        except Exception as e:
            print(f"⚠️ Could not download {symbol}: {e}")

    return test_symbols

def load_swing_data(data_dir="data/historical_swing", atr_period: int = 14, atr_mult: float = 3.0, test_start_date: str = None, test_end_date: str = None):
    print(f"Loading Swing Data from {data_dir}...")
    pattern = os.path.join(data_dir, "*_1D.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No data found in {data_dir}!")
        return None, None, None, None, None
        
    data_list = []
    price_list = []
    stop_list = []
    atr_list = []
    tickers = []
    
    max_seq_len = 0
    processed_dfs = []
    
    for f in tqdm(files, desc="Processing CSVs"):
        try:
            df = pd.read_csv(f)
            ticker = os.path.basename(f).replace("_1D.csv", "")
            
            # Standardize Columns
            df.columns = [c.lower() for c in df.columns]
            col_map = {'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}
            for k, v in col_map.items():
                if k in df.columns:
                    df[v] = df[k]
            
            # Capitalize existing full names
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col.capitalize()] = df[col]

            # Ensure Date index for filtering
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
                df.set_index('Date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            # Filter by test_start_date if provided
            if test_start_date:
                df = df[df.index >= pd.to_datetime(test_start_date)]
            
            # Filter by test_end_date if provided
            if test_end_date:
                df = df[df.index <= pd.to_datetime(test_end_date)]

            if df.empty:
                continue

            # Add Indicators
            df = add_technical_indicators(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < 10: 
                # print(f"Skipping {ticker} (len={len(df)})")
                continue
                
            processed_dfs.append((df, ticker))
            if len(df) > max_seq_len:
                max_seq_len = len(df)
                
        except Exception:
            continue
            
    if not processed_dfs:
        print("load_swing_data: No processed_dfs found! Returning 5 Nones.")
        return None, None, None, None, None
        
    print(f"Max Sequence Length: {max_seq_len}")
    
    # Second pass: Process and collect
    final_data_list = []
    
    for df, ticker in processed_dfs:
        raw_close = df['Close'].values
        raw_stop = _compute_chandelier_stop(df, atr_period=atr_period, atr_mult=atr_mult)
        
        # Simple ATR calc inline to avoid messing with imports again
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        raw_atr = tr.rolling(window=14).mean().fillna(0).values
        
        feature_cols = [
            'sma_10', 'sma_20', 'sma_50', 'sma_200', 
            'atr', 'bb_width', 'bb_upper', 'bb_lower', 
            'ema_12', 'ema_26', 
            'macd', 'macd_signal', 'macd_diff', 
            'adx', 'rsi', 'stoch_k', 'stoch_d', 
            'williams_r', 'roc', 'bb_pband', 
            'volume_sma', 'volume_ratio', 'obv', 'mfi', 'price_vs_sma20'
        ]
        
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 25:
            continue
            
        features = df[feature_cols].values
        
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        norm_features = (features - mean) / std
        norm_features = np.clip(norm_features, -10, 10)
        norm_features = np.nan_to_num(norm_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad directly here
        curr_len = len(norm_features)
        if curr_len < max_seq_len:
            pad_len = max_seq_len - curr_len
            norm_features = np.pad(norm_features, ((pad_len, 0), (0, 0)), mode='edge')
            raw_close = np.pad(raw_close, (pad_len, 0), mode='edge')
            raw_stop = np.pad(raw_stop, (pad_len, 0), mode='edge')
            raw_atr = np.pad(raw_atr, (pad_len, 0), mode='edge')
            
        data_list.append(norm_features)
        price_list.append(raw_close)
        stop_list.append(raw_stop)
        atr_list.append(raw_atr)
        tickers.append(ticker)

    if not data_list:
        return None, None, None, None, None
            
    data_tensor = torch.FloatTensor(np.array(data_list))
    price_tensor = torch.FloatTensor(np.array(price_list))
    stop_tensor = torch.FloatTensor(np.array(stop_list))
    atr_tensor = torch.FloatTensor(np.array(atr_list))
    
    return data_tensor, price_tensor, stop_tensor, atr_tensor, tickers

def run_backtest(
    model_path: str = "models/swing_best",
    test_generalization: bool = False,
    test_count: int = 40,
    period: str = "10y",
    seed: int = 1337,
    use_trailing_stop: bool = False,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    use_profit_take: bool = False,
    profit_atr_mult: float = 4.0,
    position_pct: float = 1.0,
    test_start_date: str = None,
    test_end_date: str = None,
    visualize: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Swing Bot Backtest on {device}...")
    
    data_dir = "data/historical_swing"
    if test_generalization:
        print("MODE: Generalization Test (Unseen Symbols)")
        data_dir = "data/test_swing"
        download_test_data(data_dir, test_count=test_count, period=period, interval="1d", seed=seed)
    
    # 1. Load Data
    data, prices, stop_prices, atr_values, tickers = load_swing_data(
        data_dir, 
        atr_period=atr_period, 
        atr_mult=atr_mult,
        test_start_date=test_start_date,
        test_end_date=test_end_date
    )
    if data is None:
        return
    
    print(f"Data Shape: {data.shape}")
    print(f"Tickers: {len(tickers)}")

    if test_start_date:
        print(f"Test Start Date: {test_start_date}")

    if use_trailing_stop:
        print(f"Exits: Chandelier/ATR Trailing Stop enabled (ATR={atr_period}, Mult={atr_mult})")
    if use_profit_take:
        print(f"Exits: ATR Profit-Take enabled (Profit ATR Mult={profit_atr_mult})")
    if position_pct != 1.0:
        print(f"Position Sizing: position_pct={position_pct}")
    
    # 2. Initialize Environment
    env = VectorizedTradingEnv(data, prices, device=device, position_pct=position_pct)

    # Stop series lives on same device for fast gather
    stop_prices = stop_prices.to(device)
    atr_values = atr_values.to(device)
    
    # 3. Initialize Agent
    num_features = data.shape[2]
    agent = EnsembleAgent(
        time_series_dim=num_features,
        vision_channels=num_features,
        action_dim=3,
        device=device
    )
    
    # 4. Load Model
    print(f"Loading Model from {model_path}...")
    try:
        agent.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    # 5. Run Backtest
    print("Running Backtest...")
    
    # Set to evaluation mode (no epsilon exploration)
    for sub_agent in agent.agents:
        sub_agent.epsilon = 0.0
        sub_agent.policy_net.eval()
        
    state = env.reset()
    # Force deterministic start for backtest
    env.current_step.fill_(env.window_size) 
    state = env._get_observation()
    
    # Dummy Text Data
    seq_len = 64
    dummy_text_ids = torch.zeros((env.num_envs, seq_len), dtype=torch.long).to(device)
    dummy_text_mask = torch.ones((env.num_envs, seq_len), dtype=torch.long).to(device)
    
    total_steps = env.total_steps

    total_reward = 0.0
    forced_sells = 0
    forced_profit_sells = 0
    
    # Track actions
    action_counts = {0: 0, 1: 0, 2: 0} # Hold, Buy, Sell
    all_realized_pnls = []
    
    # Visualization Setup
    viz_indices = []
    viz_data = {}
    if visualize:
        # Pick up to 3 random symbols to visualize
        import random
        num_viz = min(3, env.num_envs)
        viz_indices = random.sample(range(env.num_envs), num_viz)
        print(f"Visualization enabled for indices: {viz_indices} ({[tickers[i] for i in viz_indices]})")
        
        for idx in viz_indices:
            viz_data[idx] = {
                'prices': [],
                'actions': [],
                'portfolio_val': [], # Not easily tracked per env in vectorized, but we can track price/action
                'ticker': tickers[idx]
            }

    for step in tqdm(range(total_steps), desc="Backtesting"):
        with torch.no_grad():
            actions = agent.batch_act(state, dummy_text_ids, dummy_text_mask)

        # Optional: enforce trailing stop exits by overriding actions
        if use_trailing_stop or use_profit_take:
            row_indices = torch.arange(env.num_envs, device=device)
            current_steps = env.current_step
            current_prices = env.prices[row_indices, current_steps]
            actions = actions.clone()

            # 1) Stop-loss / trailing stop
            if use_trailing_stop:
                current_stops = stop_prices[row_indices, current_steps]
                stop_hit = env.in_position & (current_prices < current_stops)
                if stop_hit.any():
                    forced_sells += int(stop_hit.sum().item())
                    actions[stop_hit] = 2

            # 2) Profit-take: price >= entry + profit_mult * ATR
            if use_profit_take:
                current_atr = atr_values[row_indices, current_steps]
                profit_hit = env.in_position & (current_prices >= (env.entry_price + (current_atr * float(profit_atr_mult))))
                if profit_hit.any():
                    # Do not override a stop-hit sell (already sell anyway)
                    newly_forced = profit_hit & (actions != 2)
                    if newly_forced.any():
                        forced_profit_sells += int(newly_forced.sum().item())
                        actions[newly_forced] = 2
        
        # Record visualization data before step
        if visualize:
            # Correctly index prices: (num_envs,)
            row_indices = torch.arange(env.num_envs, device=device)
            current_prices_cpu = env.prices[row_indices, env.current_step].cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            for idx in viz_indices:
                viz_data[idx]['prices'].append(current_prices_cpu[idx])
                viz_data[idx]['actions'].append(actions_cpu[idx])

        next_state, rewards, dones, infos = env.step(actions)

        # Track PnL
        if 'realized_pnl' in infos:
            mask = infos.get('sell_mask')
            if mask is not None and mask.any():
                 pnl_values = infos['realized_pnl'][mask].cpu().numpy()
                 all_realized_pnls.extend(pnl_values)

        total_reward += float(rewards.sum().item())
        
        unique, counts = np.unique(actions.cpu().numpy(), return_counts=True)
        for u, c in zip(unique, counts):
            action_counts[u] += c
            
        state = next_state
        
    print("\nBacktest Results Summary:")
    print("-" * 30)
    
    # Analyze PnL
    n_trades = 0
    n_wins = 0
    if all_realized_pnls:
        pnls = np.array(all_realized_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        
        n_trades = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0
        
        avg_win = wins.mean() * 100 if n_wins > 0 else 0.0
        avg_loss = losses.mean() * 100 if n_losses > 0 else 0.0
        
        print(f"Trade Analysis (PnL):")
        print(f"  Total Trades: {n_trades}")
        print(f"  Win Rate:     {win_rate:.1f}%")
        print(f"  Avg Win:      {avg_win:.2f}%")
        print(f"  Avg Loss:     {avg_loss:.2f}%")
        if avg_loss != 0:
            print(f"  Risk/Reward:  {abs(avg_win/avg_loss):.2f}")
    else:
        print("Trade Analysis (PnL): No completed trades.")

    print(f"Action Distribution:")
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        print(f"  Hold: {action_counts.get(0, 0) / total_actions * 100:.1f}%")
        print(f"  Buy:  {action_counts.get(1, 0) / total_actions * 100:.1f}%")
        print(f"  Sell: {action_counts.get(2, 0) / total_actions * 100:.1f}%")

    if use_trailing_stop:
        print(f"Forced Sells (Trailing Stop): {forced_sells}")
    if use_profit_take:
        print(f"Forced Sells (Profit Take): {forced_profit_sells}")

    avg_reward = total_reward / max(1, env.num_envs)
    print(f"Total Reward (Sum): {total_reward:.4f}")
    print(f"Average Reward per Symbol: {avg_reward:.6f}")
    
    if visualize:
        print("\nGenerating Plots...")
        os.makedirs("plots", exist_ok=True)
        for idx, data in viz_data.items():
            ticker = data['ticker']
            prices = np.array(data['prices']).flatten()
            actions = np.array(data['actions']).flatten()
            
            print(f"   Debug: {ticker} - Prices Shape: {prices.shape}, Actions Shape: {actions.shape}")
            
            if len(prices) != len(actions):
                min_len = min(len(prices), len(actions))
                prices = prices[:min_len]
                actions = actions[:min_len]

            plt.figure(figsize=(12, 6))
            plt.plot(prices, label='Price', color='black', alpha=0.6)
            
            # Plot Buys
            buy_mask = (actions == 1)
            buy_indices = np.where(buy_mask)[0]
            buy_prices = prices[buy_mask]
            
            # Ensure sizes match (paranoid check)
            if len(buy_indices) != len(buy_prices):
                print(f"   Warning: Buy mismatch {len(buy_indices)} vs {len(buy_prices)}")
                min_len = min(len(buy_indices), len(buy_prices))
                buy_indices = buy_indices[:min_len]
                buy_prices = buy_prices[:min_len]

            if len(buy_indices) > 0:
                plt.scatter(buy_indices, buy_prices, marker='^', color='green', label='Buy', s=100, zorder=5)
            
            # Plot Sells
            sell_mask = (actions == 2)
            sell_indices = np.where(sell_mask)[0]
            sell_prices = prices[sell_mask]

            # Ensure sizes match (paranoid check)
            if len(sell_indices) != len(sell_prices):
                print(f"   Warning: Sell mismatch {len(sell_indices)} vs {len(sell_prices)}")
                min_len = min(len(sell_indices), len(sell_prices))
                sell_indices = sell_indices[:min_len]
                sell_prices = sell_prices[:min_len]
            
            if len(sell_indices) > 0:
                plt.scatter(sell_indices, sell_prices, marker='v', color='red', label='Sell', s=100, zorder=5)
            
            plt.title(f"Backtest Visualization: {ticker}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = f"plots/backtest_{ticker}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"   Saved plot to {plot_path}")

    print("\nBacktest Complete.")
    
    return {
        'total_reward': total_reward,
        'avg_reward_per_symbol': avg_reward,
        'win_rate': (n_wins / n_trades) if n_trades > 0 else 0.0,
        'trades': n_trades,
        'forced_buys': 0,
        'forced_loss_sells': forced_sells,
        'forced_profit_sells': forced_profit_sells
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing backtest + generalization check")
    parser.add_argument("model_path", nargs="?", default="models/swing_best", help="Model prefix path (no _balanced suffix)")
    parser.add_argument("--generalization", action="store_true", help="Use unseen symbols downloaded via yfinance")
    parser.add_argument("--test-count", type=int, default=40, help="How many unseen symbols to test")
    parser.add_argument("--period", type=str, default="10y", help="yfinance period (e.g. 2y, 5y, 10y)")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed for unseen symbols")
    parser.add_argument("--use-trailing-stop", action="store_true", help="Force SELL when price breaks ATR trailing stop")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period for trailing stop")
    parser.add_argument("--atr-mult", type=float, default=3.0, help="ATR multiplier for trailing stop")
    parser.add_argument("--use-profit-take", action="store_true", help="Force SELL when price reaches ATR-based profit target")
    parser.add_argument("--profit-atr-mult", type=float, default=4.0, help="Profit ATR multiplier (entry + ATR * mult)")
    parser.add_argument("--position-pct", type=float, default=1.0, help="Fraction of cash to deploy per BUY (default 1.0=all-in)")
    parser.add_argument("--test-start-date", type=str, default=None, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, default=None, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--visualize", action="store_true", help="Generate plots for a few random symbols")

    args = parser.parse_args()

    run_backtest(
        args.model_path,
        test_generalization=args.generalization,
        test_count=args.test_count,
        period=args.period,
        seed=args.seed,
        use_trailing_stop=args.use_trailing_stop,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        use_profit_take=args.use_profit_take,
        profit_atr_mult=args.profit_atr_mult,
        position_pct=args.position_pct,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        visualize=args.visualize,
    )
