# Alpaca-Bot Workspace Refactoring Plan

## Current Issues

### 1. Code Duplication
- `utils.py` and `bot_standalone.py` both define technical indicator functions
- Multiple bot files have nearly identical API connection logic
- Risk management code is scattered across files

### 2. Configuration Chaos
- API keys hardcoded in some files (`bot_money_scraper.py`, `bot_daytrader_pure.py`)
- API keys in `.env` for others (`bot_standalone.py`)
- Trading parameters scattered (MAX_POSITIONS, RISK_PER_TRADE, etc.)

### 3. Unused/Legacy Files
- `bot_money_scraper_backup.py` - backup file
- `bot_money_scraper_ws.py` - partial WebSocket experiment
- `bot_simple_force.py` - testing file
- `bot_standalone_risk_methods.py` - code fragment
- `bot_daytrader.py` vs `bot_daytrader_pure.py` - duplicate
- `bot_daytrader_crypto.py` - separate crypto bot

### 4. No Package Structure
- Everything in flat `src/` folder
- No `__init__.py` files
- Import paths are fragile

---

## Proposed New Structure

```
alpaca-bot/
├── config/
│   ├── settings.py          # Centralized config (from .env)
│   └── watchlists/
│       ├── my_portfolio.txt
│       └── day_trade_list.txt
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── indicators.py     # All technical indicators
│   │   ├── state.py          # State normalization (from utils.py)
│   │   └── risk.py           # Risk management functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── network.py        # DuelingNetwork with GRU
│   │   ├── agent.py          # DuelingDQN agent
│   │   └── buffers.py        # Replay buffers
│   ├── environments/
│   │   ├── __init__.py
│   │   └── swing_env.py      # SwingTradingEnv
│   ├── bots/
│   │   ├── __init__.py
│   │   ├── base_bot.py       # Abstract base class with shared logic
│   │   ├── money_scraper.py  # Scalper (WebSocket)
│   │   ├── swing_trader.py   # Swing trader (15min intervals)
│   │   └── day_trader.py     # Day trader
│   └── training/
│       ├── __init__.py
│       ├── trainer.py        # ParallelTrainer class
│       └── backtest.py       # Backtesting logic
├── scripts/
│   ├── train.py              # Entry point for training
│   ├── backtest.py           # Entry point for backtesting
│   └── run_bot.py            # Unified bot runner
├── models/                   # Saved model weights (.pth files)
├── data_cache/               # Cached market data
├── logs/                     # Trading logs
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
└── README.md
```

---

## Refactoring Steps

### Phase 1: Create Package Structure
1. Create `config/settings.py` - centralized configuration
2. Create `src/core/__init__.py` and sub-modules
3. Create `src/models/__init__.py` and move model code
4. Create `src/bots/base_bot.py` with shared bot logic

### Phase 2: Consolidate Code
1. Merge all indicator functions into `core/indicators.py`
2. Move risk management to `core/risk.py`
3. Extract shared API logic to `bots/base_bot.py`
4. Update imports across all files

### Phase 3: Clean Up
1. Delete unused files (backups, experiments)
2. Move watchlists to `config/watchlists/`
3. Update Dockerfile and docker-compose.yml
4. Update README.md

### Phase 4: Test
1. Run training with new structure
2. Test each bot individually
3. Verify Docker builds work

---

## Key Design Decisions

### 1. Centralized Config (`config/settings.py`)
```python
import os
from dotenv import load_dotenv

load_dotenv()

# API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Trading Parameters
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
MAX_POSITION_PCT = 0.25

# Bot-Specific Settings
class MoneyScraperConfig:
    MAX_POSITIONS = 8
    PROFIT_TARGET_PCT = 0.04
    STOP_LOSS_PCT = 0.02
    SCAN_INTERVAL = 5
    USE_WEBSOCKET = True

class SwingTraderConfig:
    MAX_POSITIONS = 1000  # Unlimited
    SCAN_INTERVAL_MINUTES = 15
    USE_WEBSOCKET = False

class DayTraderConfig:
    MAX_POSITIONS = 5
    PROFIT_TARGET_PCT = 0.01
    STOP_LOSS_PCT = 0.005
    SCAN_INTERVAL = 30
```

### 2. Base Bot Class (`bots/base_bot.py`)
```python
from abc import ABC, abstractmethod
import alpaca_trade_api as tradeapi
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL

class BaseBot(ABC):
    def __init__(self, model_path, watchlist_file):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)
        self.symbols = self.load_watchlist(watchlist_file)
        self.agent = self.load_model(model_path)
        self.position_states = {}
        
    def load_watchlist(self, filepath):
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]
    
    @abstractmethod
    def run_once(self):
        pass
    
    @abstractmethod
    def check_exits(self):
        pass
    
    def get_account_info(self):
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.daytrading_buying_power)
        }
```

---

## Files to Delete (After Backup)
- `src/bot_money_scraper_backup.py`
- `src/bot_money_scraper_ws.py`
- `src/bot_simple_force.py`
- `src/bot_standalone_risk_methods.py`
- `src/bot_daytrader.py` (keep pure version)
- `src/paper_trade.py` (merged into bots)
- `src/swing_trade.py` (legacy)
- `src/swing_train.py` (replaced by parallel_train)
- `src/full_backtest.py` (merged)
- `src/shared_backtest.py` (merged)
- `.bot_improvements_status.md`
- `.implementation_plan.md`
- `.websocket_migration_notes.md`

---

## Estimated Effort
- Phase 1: ~30 minutes
- Phase 2: ~1 hour
- Phase 3: ~15 minutes
- Phase 4: ~30 minutes

**Total: ~2-3 hours**

---

## User Approval Required

Before proceeding, please confirm:
1. ✅ Create new package structure?
2. ✅ Delete legacy/backup files?
3. ✅ Centralize all config to `.env`?
4. ✅ Start fresh training after refactor?
