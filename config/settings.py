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
    """High-frequency scalper settings"""
    MAX_POSITIONS = 8
    PROFIT_TARGET_PCT = 0.04    # +4%
    STOP_LOSS_PCT = 0.02        # -2%
    SCAN_INTERVAL_SECONDS = 5
    USE_WEBSOCKET = True
    WATCHLIST = "my_portfolio.txt"

class SwingTraderConfig:
    """Multi-day swing trader settings"""
    MAX_POSITIONS = 1000        # Virtually unlimited
    SCAN_INTERVAL_MINUTES = 15
    USE_WEBSOCKET = False
    WATCHLIST = "my_portfolio.txt"
    # ATR-based risk (matches training environment)
    STOP_ATR_MULT = 2.5
    TRAILING_ATR_MULT = 3.0
    PROFIT_ATR_MULT = 4.0

class DayTraderConfig:
    """Intraday scalper settings"""
    MAX_POSITIONS = 5
    PROFIT_TARGET_PCT = 0.005   # +0.5%
    STOP_LOSS_PCT = 0.003       # -0.3%
    SCAN_INTERVAL_SECONDS = 30
    USE_WEBSOCKET = False
    WATCHLIST = "day_trade_list.txt"

# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_DIR = Path(__file__).parent.parent / "models"
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
    BATCH_SIZE = 64
    EPISODES_PER_SYMBOL = 10
    
    @classmethod
    def get_state_size(cls):
        return (cls.WINDOW_SIZE * cls.NUM_WINDOW_FEATURES) + cls.NUM_REGIME_FEATURES + cls.NUM_PORTFOLIO_FEATURES
