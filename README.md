# Alpaca Trading Bot 🦙

An AI-powered trading system for Alpaca Markets, featuring Swing Trading, Day Trading, and Scalping bots powered by a Transformer-based Dueling DQN agent.

## 🌟 Features
- **4 Specialized Bots**:
  - `Money Scraper`: High-frequency scalper (WebSocket) for quick gains.
  - `Swing Trader`: Multi-day position trader using ATR-based risk management.
  - `Day Trader`: Intraday scalper with tight stops.
  - `Crypto Trader`: 24/7 crypto trader (spot) using the same agent interface.
- **Advanced AI Model**: Transformer-based Dueling DQN with optional NoisyNet exploration.
- **Robust Risk Management**: ATR trailing stops, position sizing logic, and portfolio concentration limits.
- **Dockerized**: Easy deployment with Docker Compose.

## 📂 Project Structure
```
alpaca-bot/
├── config/             # Centralized configuration & Watchlists
├── src/
│   ├── bots/           # Bot implementations
│   ├── core/           # Shared logic (Indicators, Risk, State)
│   ├── models/         # Neural Network & Agent code
│   └── training/       # Training pipeline
├── scripts/            # Entry points (Run bots, Train, Backtest)
└── models/             # Saved model weights
```

## 🚀 Getting Started

### 1. Setup
Create a `.env` file with your Alpaca API keys (see `.env.example`).
```ini
SWING_API_KEY=...
SWING_API_SECRET=...
# ... (See config/settings.py for full list)

# Paper trading (recommended)
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Crypto Trader (optional)
CRYPTO_API_KEY=...
CRYPTO_API_SECRET=...
# Optional tuning
# CRYPTO_ORDER_NOTIONAL=25
# CRYPTO_WATCHLIST=crypto_watchlist.txt
```

### 2. Train the Model
Start GPU training (logs to `logs/training.log`):
```bash
python manage.py train
```

Check progress:
```powershell
Get-Content logs\training.log -Wait
```

### 3. Run a Bot
Run via Docker Compose (recommended) or directly with the scripts in `src/bots/`.

Run only the crypto trader:
```bash
docker compose up --build -d cryptotrader
```

Or run directly:
```bash
python -m src.bots.crypto_trader
```

## ✅ Current Entrypoints
- **Train (current)**: `python manage.py train` → runs `scripts/train_gpu.py` (scalper V2 pipeline).
- **Status/Stop**: `python manage.py status` / `python manage.py stop`
- **Bots (current)**: `docker-compose up --build -d`

## 🧰 Legacy / Experimental
These exist for reference or older experiments and are not part of the default workflow:
- `scripts/train_gpu_swing.py`, `scripts/backtest_swing.py`
- `src/training/trainer.py` (parallel CPU trainer)

### 4. Deploy with Docker (Recommended)
Build and start all bots:
```bash
docker-compose up --build -d
```
To view logs:
```bash
docker-compose logs -f money_scraper
```

## 🧠 Model Architecture
The agent uses a **Transformer-based Dueling DQN**:
1. **Input**: $\text{WINDOW\_SIZE}$-bar window of features (default: 60) flattened for the network.
2. **Transformer Encoder**: Multi-head self-attention over the time-series window.
3. **Dueling Heads**: Separates Value $V(s)$ and Advantage $A(s,a)$ streams.
4. **Actions**: Hold, Buy, Sell.

## ⚠️ Disclaimer
This software is for educational purposes only. Do not risk money you cannot afford to lose. Use at your own risk.