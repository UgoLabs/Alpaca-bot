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


class MoneyScraperCreds:
    API_KEY = os.getenv("SCRAPER_API_KEY")
    API_SECRET = os.getenv("SCRAPER_API_SECRET")


class DayTraderCreds:
    API_KEY = os.getenv("DAY_API_KEY")
    API_SECRET = os.getenv("DAY_API_SECRET")


class CryptoTraderCreds:
    # Prefer crypto-specific vars, but support Alpaca's conventional env names.
    API_KEY = os.getenv("CRYPTO_API_KEY") or os.getenv("APCA_API_KEY_ID")
    API_SECRET = os.getenv("CRYPTO_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")


# Shared
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# Options bot always talks to this host (defaults to paper even if ALPACA_BASE_URL is live).
OPTIONS_ALPACA_BASE_URL = os.getenv(
    "OPTIONS_ALPACA_BASE_URL",
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


class MoneyScraperConfig:
    """High-frequency scalper (The 'Crumbs')"""
    MAX_POSITIONS = 10
    PROFIT_TARGET_PCT = 0.003   # +0.3% (Quick Scalp)
    STOP_LOSS_PCT = 0.0015      # -0.15% (Tight Stop)
    SCAN_INTERVAL_SECONDS = 5
    USE_WEBSOCKET = True
    WATCHLIST = "my_portfolio.txt"
    LIQUIDATE_EOD = True
    ONLINE_LEARNING = True
    EXPLORATION_EPSILON = 0.05  # 5% random exploration for trade discovery


class SwingTraderConfig:
    WATCHLIST = "swing_liquid.txt"  # ~500 liquid names (build via scripts/build_liquid_watchlist.py)
    MAX_POSITIONS = 40           # Portfolio sweep (2024+ OOS): 40 slots beat 20/30 on return & Sharpe
    PROFIT_TARGET_PCT = 0.20    # Only used if no Stop ATR
    STOP_LOSS_PCT = 0.10        # Only used if no Stop ATR
    SCAN_INTERVAL_MINUTES = 60
    CONFIDENCE_THRESHOLD = 0.70 # Portfolio sweep: 0.70 + slot ranking -> +68.9% vs SPY +18.8%
                                # (alpha +50%), Sharpe 1.66, max DD -13.3% at 40 slots.
    SELL_CONFIDENCE_THRESHOLD = 0.35 # Portfolio sell sweep (40 slots, buy 0.70): best Sharpe 2.12,
                                     # +97.7% vs SPY +56.8%, max DD -13.1%, win 80%

    # Hold winners longer (daily bars): defer agent SELL until trail or min hold
    ENABLE_WINNER_TRAIL = False  # Liquid OOS: trail off beat tuned trail on (+55% vs ~+38%, Sharpe 2.15)
    MIN_HOLD_DAYS = 2              # Ignore soft agent SELL for first N sessions (hard stop still applies)
    TRAIL_ACTIVATION_PCT = 0.05    # Start trailing once position is up +5%
    TRAIL_GIVEBACK_PCT = 0.03      # Exit when price gives back 3% from peak unrealized gain
    
    # Enable Trailing Stops (Defaults)
    USE_TRAILING_STOP = True
    STOP_ATR_MULT = 2.0 # Margin Safe (Tight Stops)
    PROFIT_ATR_MULT = 5.0
    TRAILING_ATR_MULT = 100.0 # Disable strict trailing stop (still rely on Agent + Hard Stop)
    PROFIT_ATR_MULT = 5.0     # Profit Target (5x ATR)
    LIQUIDATE_EOD = False
    ONLINE_LEARNING = True
    EXPLORATION_EPSILON = 0.03  # 3% exploration for swing trades
    DATA_FEED = 'iex'  # Free real-time data (fractional market volume)
    DATA_MAX_AGE_HOURS = 20       # Only re-download CSVs older than this on startup
    WAKE_BEFORE_OPEN_MINUTES = 30 # Wake this many minutes before open to finish CSV refresh
    PREMARKET_DOWNLOAD_WORKERS = 4  # Parallel yfinance workers during pre-open refresh
    INFER_BATCH_SIZE = 128          # GPU batch size for live scoring (matches backtest batch_act)
    ENABLE_SPY_FEAR_FILTER = True   # Block new buys when broad market is down (see SPY_FEAR_BLOCK_PCT)
    SPY_FEAR_BLOCK_PCT = -1.0       # vs prior close; set ENABLE_SPY_FEAR_FILTER=False to disable
    VERBOSE_SCAN = False            # If True, log every symbol action (slow with large watchlists)
    CASH_ONLY = True              # Never use margin: size orders from cash on hand only
    POSITION_SIZE_PCT = 1.0       # Fraction of equal-weight slot to deploy (1.0 = full slot)
    CASH_BUFFER_PCT = 0.02        # Keep 2% of equity undeployed (fees / rounding buffer)


class OptionsTraderConfig:
    """Swing-signal options bot: call debit spreads, paper only, full liquid universe.

    Credentials: DAY_API_KEY / DAY_API_SECRET (same account as day trader).
    """

    WATCHLIST = "options_liquid_200.txt"  # Live scan (top 200 $ vol; matches training)
    TRAIN_WATCHLIST = "options_liquid_200.txt"  # Options fine-tune / marks download
    MAX_POSITIONS = 40               # 200-symbol backtest: 40 slots @ conf 0.80 (+26.4% OOS)
    CONFIDENCE_THRESHOLD = 0.80      # Stricter entries on 200-name book (0.70 was for 20-name list)
    SELL_CONFIDENCE_THRESHOLD = 0.35
    MIN_HOLD_DAYS = 2
    ENABLE_WINNER_TRAIL = False
    TRAIL_ACTIVATION_PCT = 0.05
    TRAIL_GIVEBACK_PCT = 0.03
    ENABLE_SPY_FEAR_FILTER = True
    SPY_FEAR_BLOCK_PCT = -1.0
    CASH_ONLY = False                # Options only: size from buying power (margin allowed)
    POSITION_SIZE_PCT = 1.0
    CASH_BUFFER_PCT = 0.02           # Buffer applied to equity-based slot size
    MIN_BUYING_POWER = 50.0          # Skip new spreads if BP below this
    SCAN_INTERVAL_SECONDS = 300      # Same default as swing live loop (5 min)
    DATA_FEED = "iex"
    DATA_MAX_AGE_HOURS = 20
    WAKE_BEFORE_OPEN_MINUTES = 90       # 200 symbols: CSV + option marks + snapshots + features
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
    SPREAD_WIDTH = 5.0               # Strike width ($) for call debit spread
    LIMIT_SLIPPAGE_PCT = 0.08        # Pay up to 8% above estimated debit for fills
    PREMIUM_STOP_PCT = 0.40          # Close spread if aggregate option P/L <= -40%
    MIN_DTE_EXIT = 5                 # Close before expiry week


class DayTraderConfig:
    """Intraday trader on 15-minute bars (real-time IEX base data).

    Default: strict rules-only simple trend follow (no RL). Probe-aligned holds:
    cut losers fast, ~1.5–3h horizon, EOD flat.
    """
    RULES_ONLY = False               # Set True for rules-only trend; False for RL (day_from_swing)
    DAY_FEATURE_SET = "swing"        # 'swing' (11, transfer) | 'day' (15 intraday features)
    TREND_VARIANT = "simple"         # simple | ema_cross | ma_stack | vwap
    EXIT_ON_TREND_BREAK = True       # Flat when price loses trend structure

    MAX_POSITIONS = 10
    CONFIDENCE_THRESHOLD = 0.65      # ep23 OOS sweep best (+0.14% watchlist vs +0.10% raw)
    SELL_CONFIDENCE_THRESHOLD = 0.30 # Unused in rules-only exits

    # --- SPY regime (block new longs when broad market is weak) ---
    ENABLE_SPY_TREND_FILTER = True   # Require SPY above session VWAP + bullish EMAs
    SPY_VWAP_MIN_DIST = 0.0          # Min session VWAP distance (0 = at or above VWAP)
    ENABLE_SPY_FEAR_FILTER = True    # Also block when SPY day change < SPY_FEAR_BLOCK_PCT
    SPY_FEAR_BLOCK_PCT = -0.5        # vs prior close (%); tighter than swing -1.0

    # --- Risk + hold horizon (15Min bars) ---
    STOP_LOSS_PCT = 0.006        # -0.6% hard stop
    MIN_HOLD_BARS = 8            # ~2h before soft exits on winners
    MAX_HOLD_BARS = 20           # ~5h time stop (longer trend follow)
    ENABLE_WINNER_TRAIL = False  # Prefer min/max hold over tight trail
    TRAIL_ACTIVATION_PCT = 0.015 # Only if ENABLE_WINNER_TRAIL=True
    TRAIL_GIVEBACK_PCT = 0.008
    PROFIT_TARGET_PCT = 0.08     # Absolute ceiling for parabolic spikes (+8%)
    COST_PER_SIDE = 0.0005       # 5 bps/side for backtest planning

    SCAN_INTERVAL_SECONDS = 300  # Re-check positions/signals every 5 min (bars are 15Min)
    USE_WEBSOCKET = False
    WATCHLIST = "day_trade_list.txt"
    LIQUIDATE_EOD = True
    ONLINE_LEARNING = False      # No RL updates when RULES_ONLY=True
    EXPLORATION_EPSILON = 0.0
    DATA_FEED = 'iex'  # Real-time IEX (free tier). Liquid names only.



class CryptoTraderConfig:
    """24/7 crypto trader settings."""
    MAX_POSITIONS = int(os.getenv("CRYPTO_MAX_POSITIONS", "5"))
    CONFIDENCE_THRESHOLD = 0.60
    SCAN_INTERVAL_SECONDS = int(os.getenv("CRYPTO_SCAN_INTERVAL_SECONDS", "30"))
    WATCHLIST = os.getenv("CRYPTO_WATCHLIST", "crypto_watchlist.txt")
    DATA_FEED = 'sip'  # Default, though crypto uses get_crypto_bars
    # Crypto tends to be noisier; default to a bit more exploration.
    EXPLORATION_EPSILON = float(os.getenv("CRYPTO_EXPLORATION_EPSILON", "0.08"))
    # If set, will attempt to close positions when stopping.
    LIQUIDATE_ON_EXIT = os.getenv("CRYPTO_LIQUIDATE_ON_EXIT", "0") in ("1", "true", "TRUE", "yes", "YES")

# =============================================================================
# MODEL PATHS
# =============================================================================


MODEL_DIR = Path(__file__).parent.parent / "models"

# Specialized Model Paths (Option A Architecture)
SWING_MODEL_PATH = MODEL_DIR / "swing_gen7_refined_ep380_balanced.pth"  # Gen 7 EP380 (Stop 3.0/Profit 5.0 Winner)
OPTIONS_MODEL_PATH = MODEL_DIR / "options_from_swing_200_ep50_balanced.pth"  # 200-name OOS: 40 slots @ 0.80
SCALPER_MODEL_PATH = MODEL_DIR / "day_from_swing_ep23_balanced.pth"  # pinned ep23 (+0.10% OOS watchlist)
DAY_FROM_SWING_MODEL_PREFIX = "models/day_from_swing"  # train_day_phase2 --init-from-swing output
# If True, options mode loads SWING_MODEL_PATH instead of OPTIONS_MODEL_PATH
OPTIONS_USE_SWING_MODEL = False
SHARED_MODEL_PATH = MODEL_DIR / "swing_best_balanced.pth"  # Using Gen 5 for Crypto
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

    # Action-shaping knobs (defaults preserve the original swing/crypto behavior).
    # Intraday (15Min) training overrides these in scripts/train_day_phase2.py:
    #   - FLAT_DOWNTREND_BONUS must be ~0 intraday or the agent learns to never buy
    #     (it gets paid every step it sits in cash during dips).
    #   - INVALID_ACTION_PENALTY too small lets the agent spam SELL as a free no-op.
    FLAT_DOWNTREND_BONUS = float(os.getenv("FLAT_DOWNTREND_BONUS", "0.01"))
    INVALID_ACTION_PENALTY = float(os.getenv("INVALID_ACTION_PENALTY", "0.000001"))

    # Entry-quality reward (forward-return shaping). 0 = disabled (default).
    # Intraday training enables this so the agent learns profitable entries
    # instead of collapsing to a never-buy policy.
    ENTRY_REWARD_COEF = float(os.getenv("ENTRY_REWARD_COEF", "0.0"))
    ENTRY_LOOKAHEAD_BARS = int(os.getenv("ENTRY_LOOKAHEAD_BARS", "6"))
    # Intraday fine-tune: force exit after N bars (0 = disabled). 20 bars @ 15Min ~ 5h.
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
