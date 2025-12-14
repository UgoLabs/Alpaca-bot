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

# Shared
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# =============================================================================
# TRADING PARAMETERS (Shared)
# =============================================================================

RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.25))

# =============================================================================
# BOT-SPECIFIC SETTINGS
# =============================================================================

class MoneyScraperConfig:
    """High-frequency scalper (The 'Crumbs')"""
    MAX_POSITIONS = 8
    PROFIT_TARGET_PCT = 0.003   # +0.3% (Quick Scalp)
    STOP_LOSS_PCT = 0.0015      # -0.15% (Tight Stop)
    SCAN_INTERVAL_SECONDS = 5
    USE_WEBSOCKET = True
    WATCHLIST = "my_portfolio.txt"
    LIQUIDATE_EOD = True
    ONLINE_LEARNING = True

class SwingTraderConfig:
    """Multi-day swing trader settings"""
    MAX_POSITIONS = 1000
    SCAN_INTERVAL_MINUTES = 15
    USE_WEBSOCKET = False
    WATCHLIST = "my_portfolio.txt"
    STOP_ATR_MULT = 2.5
    TRAILING_ATR_MULT = 3.0
    PROFIT_ATR_MULT = 4.0
    LIQUIDATE_EOD = False
    ONLINE_LEARNING = True

class DayTraderConfig:
    """Intraday trend trader (The 'Slice')"""
    MAX_POSITIONS = 5
    PROFIT_TARGET_PCT = 0.015   # +1.5% (Day Trend)
    STOP_LOSS_PCT = 0.005       # -0.5% (Room to breathe)
    SCAN_INTERVAL_SECONDS = 30
    USE_WEBSOCKET = False
    WATCHLIST = "day_trade_list.txt"
    LIQUIDATE_EOD = True
    ONLINE_LEARNING = True

# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_DIR = Path(__file__).parent.parent / "models"

# Specialized Model Paths (Option A Architecture)
SWING_MODEL_PATH = MODEL_DIR / "swing_dqn_best.pth"      # Daily bars - Swing Trader
SCALPER_MODEL_PATH = MODEL_DIR / "scalper_dqn_best.pth"  # 5Min bars - Money Scraper & Day Trader

# Legacy (fallback)
SHARED_MODEL_PATH = MODEL_DIR / "SHARED_dqn_best.pth"
REPLAY_BUFFER_PATH = MODEL_DIR / "replay_buffer.pkl"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

class TrainingConfig:
    WINDOW_SIZE = 20
    NUM_WINDOW_FEATURES = 11
    NUM_REGIME_FEATURES = 6
    NUM_PORTFOLIO_FEATURES = 5
    GAMMA = 0.99
    LEARNING_RATE = 0.0003
    BATCH_SIZE = 512
    EPISODES_PER_SYMBOL = 10
    
    @classmethod
    def get_state_size(cls):
        return (cls.WINDOW_SIZE * cls.NUM_WINDOW_FEATURES) + cls.NUM_REGIME_FEATURES + cls.NUM_PORTFOLIO_FEATURES
