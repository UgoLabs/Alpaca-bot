"""
Centralized Configuration for Alpaca Trading Bots
All API keys and trading parameters loaded from .env
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =============================================================================
# API CREDENTIALS (Three Separate Accounts)
# =============================================================================


class SwingTraderCreds:
    API_KEY = os.getenv("SWING_API_KEY")
    API_SECRET = os.getenv("SWING_API_SECRET")


class OptionsTraderCreds:
    """Options spread bot (paper by default). Falls back to legacy DAY_* env names."""
    API_KEY = os.getenv("OPTIONS_API_KEY") or os.getenv("DAY_API_KEY")
    API_SECRET = os.getenv("OPTIONS_API_SECRET") or os.getenv("DAY_API_SECRET")


class PaperSwingTraderCreds:
    """Paper mirror of live swing: same bot, PAPER_SWING_* keys + paper API host."""
    API_KEY = os.getenv("PAPER_SWING_API_KEY")
    API_SECRET = os.getenv("PAPER_SWING_API_SECRET")


# Shared
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# Options bot always talks to this host (defaults to paper even if ALPACA_BASE_URL is live).
OPTIONS_ALPACA_BASE_URL = os.getenv(
    "OPTIONS_ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
)
PAPER_SWING_ALPACA_BASE_URL = os.getenv(
    "PAPER_SWING_ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
)

# =============================================================================
# TRADING PARAMETERS (Shared)
# =============================================================================

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))

# =============================================================================
# BOT-SPECIFIC SETTINGS
# =============================================================================


class SwingTraderConfig:
    """Daily swing equity bot: Gen7 signals, agent-gated exits (see backtest_swing_portfolio.py).

    Live exits: SELL when model confidence > SELL_CONFIDENCE_THRESHOLD. No hard stop
    and no winner trail in production (both off OOS on the liquid book).
    """

    WATCHLIST = "swing_liquid.txt"  # ~500 liquid names (build via scripts/build_liquid_watchlist.py)
    MAX_POSITIONS = 40           # Portfolio sweep (2024+ OOS): 40 slots beat 20/30 on return & Sharpe
    # 0.50/0.35 on GPU (2024-01-01+, swing_liquid, 10bps/side): +93.5%, Sharpe 2.56, 320 trades.
    # --cpu inflates to ~+101%; use GPU for backtest/sweep (matches live bot inference).
    # 0.70 (prior live) -> ~+53%, underperforms 0.50 buy on current data.
    CONFIDENCE_THRESHOLD = 0.50
    SELL_CONFIDENCE_THRESHOLD = 0.35  # Sell sweep (40 slots): best Sharpe at 0.35; unchanged in buy re-sweep

    # Winner trail (off in prod; set True to defer soft SELLs on winners)
    ENABLE_WINNER_TRAIL = False  # Liquid OOS: trail off beat tuned trail on (+55% vs ~+38%, Sharpe 2.15)
    MIN_HOLD_DAYS = 2            # When trail on: ignore soft agent SELL for first N sessions on winners
    TRAIL_ACTIVATION_PCT = 0.05  # When trail on: start trailing once position is up +5%
    TRAIL_GIVEBACK_PCT = 0.03    # When trail on: exit when price gives back 3% from peak gain

    SCAN_INTERVAL_MINUTES = 60
    SCAN_INTERVAL_SECONDS = 3600   # 60 min between scans (1D bars; was falling back to 300s)
    LIQUIDATE_EOD = False
    DATA_FEED = 'iex'  # Free real-time data (fractional market volume)
    DATA_MAX_AGE_HOURS = 20       # Only re-download CSVs older than this on startup
    WAKE_BEFORE_OPEN_MINUTES = 30 # Wake this many minutes before open to finish CSV refresh
    PREMARKET_DOWNLOAD_WORKERS = 4  # Parallel yfinance workers during pre-open refresh
    INFER_BATCH_SIZE = 128          # GPU batch size for live scoring (matches backtest batch_act)
    ENABLE_SPY_FEAR_FILTER = False  # TEMP: off so session can open buys on red SPY days
    SPY_FEAR_BLOCK_PCT = -1.0       # vs prior close; re-enable ENABLE_SPY_FEAR_FILTER when done testing
    VERBOSE_SCAN = False            # If True, log every symbol action (slow with large watchlists)
    CASH_ONLY = True              # Never use margin: size orders from cash on hand only
    POSITION_SIZE_PCT = 1.0       # Fraction of equal-weight slot to deploy (1.0 = full slot)
    CASH_BUFFER_PCT = 0.02        # Keep 2% of equity undeployed (fees / rounding buffer)


class OptionsTraderConfig:
    """Swing-signal options bot: multi-strategy spreads (paper only), full liquid universe.

    Credentials: OPTIONS_API_KEY / OPTIONS_API_SECRET (legacy: DAY_API_*).
    """

    WATCHLIST = "swing_liquid.txt"  # Full 500-name liquid universe (parity with swing bot)
    TRAIN_WATCHLIST = "swing_liquid.txt"  # Options marks download + training universe
    MAX_POSITIONS = 40
    # 500-symbol OOS (Feb 2024+, swing_liquid, SPY filter, eps06_best, 494/500 marks):
    # 0.60/0.50 on GPU: +72.2%, Sharpe 0.78, max DD -38.1%, 1286 opens (backtest = sweep w/o --cpu).
    # Sweep with --cpu inflates to ~+103%; do not use --cpu for threshold research.
    # Prior prod (200 sym, 0.65/0.35): +48.4%.
    CONFIDENCE_THRESHOLD = 0.60
    SELL_CONFIDENCE_THRESHOLD = 0.50
    # Softmax temperature for the confidence value ONLY (action selection is unchanged).
    # Q-gap measurement (scripts/measure_q_gap.py, ep30 BUY picks): median top1-top2 Q gap
    # ~0.0066, so at temp=0.01 BUY confidence already spans ~0.37-0.85 (p25-p90 0.50-0.75),
    # 0% saturation. With only 3 actions, RAISING temp collapses confidence toward the 0.333
    # uniform floor (>=0.05 makes any threshold >~0.45 reject all buys). So 0.01 is correct;
    # the real lever is CONFIDENCE_THRESHOLD. Swing omits this attr and keeps 0.01 too.
    CONFIDENCE_TEMPERATURE = 0.01
    MIN_HOLD_DAYS = 2
    ENABLE_WINNER_TRAIL = False
    TRAIL_ACTIVATION_PCT = 0.05
    TRAIL_GIVEBACK_PCT = 0.03
    ENABLE_SPY_FEAR_FILTER = True  # OOS grid: +46.2% vs +41.8% baseline, DD -31.9% vs -34.6%
    SPY_FEAR_BLOCK_PCT = -1.0
    CASH_ONLY = False                # Options only: size from buying power (margin allowed)
    POSITION_SIZE_PCT = 1.0
    CASH_BUFFER_PCT = 0.02           # Buffer applied to equity-based slot size
    MIN_BUYING_POWER = 50.0          # Skip new spreads if BP below this
    SCAN_INTERVAL_SECONDS = 3600      # Same default as swing live loop (5 min)
    DATA_FEED = "iex"
    DATA_MAX_AGE_HOURS = 20
    WAKE_BEFORE_OPEN_MINUTES = 90       # 500 symbols: CSV + option marks + snapshots + features
    PREMARKET_DOWNLOAD_WORKERS = 6
    PREMARKET_FETCH_WORKERS = 10        # Parallel fetch_and_process before open
    PREMARKET_ONLY_FETCH = True         # No bulk download/fetch during session scans
    PREMARKET_OPTIONS_BARS_UPDATE = True  # Incremental OCC/marks refresh before open
    INFER_BATCH_SIZE = 128
    VERBOSE_SCAN = False
    LIQUIDATE_EOD = False
    ONLINE_LEARNING = False
    EXPLORATION_EPSILON = 0.0
    # --- Options-specific ---
    TARGET_DTE = 30
    MIN_DTE = 14
    MAX_DTE = 45
    SPREAD_WIDTH = 5.0               # Base strike width ($); scaled by price (see SCALE_WIDTH_BY_PRICE)
    SCALE_WIDTH_BY_PRICE = True      # <$100 -> 2.5, $100-250 -> base, >$250 -> 2x base (backtest parity)
    LIMIT_SLIPPAGE_PCT = 0.08        # Pay up to 8% above estimated debit for fills
    PREMIUM_STOP_PCT = 0.40          # Close spread if aggregate option P/L <= -40%
    PROFIT_TARGET_PCT = 0.50         # Take profit: close spread once option P/L >= +50% of premium
    MIN_DTE_EXIT = 5                 # Close before expiry week
    CLOSE_USE_MARKET = True          # Close spreads with a market MLEG order (reliable fills > limit 0.01)
    MIN_OPEN_INTEREST = 10           # Skip contracts with open interest below this (liquidity)
    MAX_CONTRACTS_PER_SLOT = 10      # Cap contracts per spread so one name can't hog the slot
    # --- Multi-strategy (live paper) ---
    # Bullish model BUY: try in order until one fits the slot budget
    BULLISH_STRATEGIES = ("call_debit", "bull_put_credit", "long_call")
    LONG_CALL_MIN_CONFIDENCE = 0.70  # long_call only when BUY conf >= this
    # Bearish opens use a SEPARATE model (BUY = open bearish); main model SELL only closes.
    ENABLE_BEARISH_OPENS = False  # enable after training options_bearish_from_swing
    BEARISH_STRATEGIES = ("put_debit", "bear_call_credit")
    BEARISH_CONFIDENCE_THRESHOLD = 0.65


# =============================================================================
# MODEL PATHS
# =============================================================================


MODEL_DIR = Path(__file__).parent.parent / "models"

# Specialized Model Paths (Option A Architecture)
SWING_MODEL_PATH = MODEL_DIR / "swing_gen7_refined_ep380_balanced.pth"  # Gen 7 EP380 (Stop 3.0/Profit 5.0 Winner)
OPTIONS_MODEL_PATH = MODEL_DIR / "options_unified_gen380_eps06_best_balanced.pth"  # unified 5-strat; OOS +48.1% @ 0.65/0.35 (200 sym)
OPTIONS_BEARISH_MODEL_PATH = MODEL_DIR / "options_bearish_from_swing_best_balanced.pth"  # separate bearish head
# If True, options mode loads SWING_MODEL_PATH instead of OPTIONS_MODEL_PATH
OPTIONS_USE_SWING_MODEL = False
REPLAY_BUFFER_PATH = MODEL_DIR / "replay_buffer.pkl"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================


class TrainingConfig:
    # Lookback window (bars). Default 60, override with WINDOW_SIZE env var.
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))
    NUM_WINDOW_FEATURES = 11
    NUM_REGIME_FEATURES = 6
    NUM_PORTFOLIO_FEATURES = 5
    
    # === LIVE TRADING ALIGNMENT ===
    # These match SwingTraderConfig for consistent training/deployment
    MAX_POSITIONS = 10
    STOP_ATR_MULT = 6.0       # Stop loss at 6x ATR
    PROFIT_ATR_MULT = 3.0     # Take profit at 3x ATR 
    CONFIDENCE_THRESHOLD = 0.60  # Only take trades with >60% softmax confidence (100% backtest win rate)
    
    # Enable ATR-based stops during training (matches live trading)
    USE_TRAILING_STOP = True
    TRAILING_STOP_ATR_MULT = 6.0  # Same as STOP_ATR_MULT
    USE_PROFIT_TAKE = True
    PROFIT_TAKE_ATR_MULT = 3.0    # Same as PROFIT_ATR_MULT
    ATR_WINDOW = 14
    
    # Execution realism (basis points)
    # 7 bps transaction cost + 3 bps slippage = 10 bps total (0.10%) per trade
    # This discourages frivolous trading and encourages meaningful entries/exits
    TRANSACTION_COST_BPS = float(os.getenv("TRANSACTION_COST_BPS", "7.0"))
    SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "3.0"))
    
    # Exit Reward Shaping (new)
    EXIT_PROFIT_BONUS = float(os.getenv("EXIT_PROFIT_BONUS", "0.001"))  # Bonus for profitable exits
    HOLDING_LOSS_PENALTY = float(os.getenv("HOLDING_LOSS_PENALTY", "0.0001"))  # Penalty per step holding a loser
    LOSS_THRESHOLD_PCT = float(os.getenv("LOSS_THRESHOLD_PCT", "0.02"))  # -2% triggers holding penalty

    FLAT_DOWNTREND_BONUS = float(os.getenv("FLAT_DOWNTREND_BONUS", "0.01"))
    INVALID_ACTION_PENALTY = float(os.getenv("INVALID_ACTION_PENALTY", "0.000001"))
    ENTRY_REWARD_COEF = float(os.getenv("ENTRY_REWARD_COEF", "0.0"))
    ENTRY_LOOKAHEAD_BARS = int(os.getenv("ENTRY_LOOKAHEAD_BARS", "6"))
    MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "0"))
    
    # === REGIME-AWARE REWARD SHAPING ===
    # Instead of oversampling bear markets (which 3x slows training), we give
    # bonus rewards for correct actions during downtrends. This teaches the model
    # to handle bear markets without increasing training data size.
    REGIME_REWARD_MULT = 2.0     # Multiplier for bear market rewards
    TREND_LOOKBACK = 20          # Days to detect downtrend (price < 95% of lookback price)
    
    # Reward scaling for better Q-value spread
    # Log-returns are tiny (-0.001 to +0.001). Scale to (-0.1 to +0.1) for better gradients.
    REWARD_SCALE = 100.0
    
    GAMMA = 0.999  # Increased from 0.99 for long-term focus
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 512
    EPISODES_PER_SYMBOL = 10
    
    @classmethod
    def get_state_size(cls):
        return (cls.WINDOW_SIZE * cls.NUM_WINDOW_FEATURES) + cls.NUM_REGIME_FEATURES + cls.NUM_PORTFOLIO_FEATURES
