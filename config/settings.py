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
    """Multi-day swing trader settings"""
    MAX_POSITIONS = 20
    SCAN_INTERVAL_MINUTES = int(os.getenv("SWING_SCAN_INTERVAL_MINUTES", "5"))
    USE_WEBSOCKET = False
    WATCHLIST = "my_portfolio.txt"
    # Tuned exits (Phase 2): mandatory ATR trailing stop + ATR profit-take
    # Increased to 6.0 for Episode 330 "Aggressive Update" Model (Catastrophe Insurance only)
    STOP_ATR_MULT = 6.0
    TRAILING_ATR_MULT = 6.0
    PROFIT_ATR_MULT = 3.0
    LIQUIDATE_EOD = False
    ONLINE_LEARNING = True
    EXPLORATION_EPSILON = 0.03  # 3% exploration for swing trades
    DATA_FEED = 'sip'  # 15-min delayed data (if no paid sub)


class DayTraderConfig:
    """Intraday trend trader (The 'Slice')"""
    MAX_POSITIONS = 10
    PROFIT_TARGET_PCT = 0.015   # +1.5% (Day Trend)
    STOP_LOSS_PCT = 0.005       # -0.5% (Room to breathe)
    SCAN_INTERVAL_SECONDS = 900 # 15 minutes
    USE_WEBSOCKET = False
    WATCHLIST = "my_portfolio.txt"  # Expanded from day_trade_list.txt
    LIQUIDATE_EOD = True
    ONLINE_LEARNING = True
    EXPLORATION_EPSILON = 0.05  # 5% exploration for day trades
    DATA_FEED = 'iex'  # Free real-time data


class CryptoTraderConfig:
    """24/7 crypto trader settings."""
    MAX_POSITIONS = int(os.getenv("CRYPTO_MAX_POSITIONS", "5"))
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
SWING_MODEL_PATH = MODEL_DIR / "swing_gen6_finetune_aggressive_update_ep330_balanced.pth"  # Aggressive Update Ep 330 (Record High Profit)
SCALPER_MODEL_PATH = MODEL_DIR / "swing_best_balanced.pth"  # Using Gen 5 for Day
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
    # Execution realism (basis points)
    # 7 bps transaction cost + 3 bps slippage = 10 bps total (0.10%) per trade
    # This discourages frivolous trading and encourages meaningful entries/exits
    TRANSACTION_COST_BPS = float(os.getenv("TRANSACTION_COST_BPS", "7.0"))
    SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "3.0"))
    
    # Exit Reward Shaping (new)
    EXIT_PROFIT_BONUS = float(os.getenv("EXIT_PROFIT_BONUS", "0.001"))  # Bonus for profitable exits
    HOLDING_LOSS_PENALTY = float(os.getenv("HOLDING_LOSS_PENALTY", "0.0001"))  # Penalty per step holding a loser
    LOSS_THRESHOLD_PCT = float(os.getenv("LOSS_THRESHOLD_PCT", "0.02"))  # -2% triggers holding penalty
    GAMMA = 0.999  # Increased from 0.99 for long-term focus
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 512
    EPISODES_PER_SYMBOL = 10
    
    @classmethod
    def get_state_size(cls):
        return (cls.WINDOW_SIZE * cls.NUM_WINDOW_FEATURES) + cls.NUM_REGIME_FEATURES + cls.NUM_PORTFOLIO_FEATURES
