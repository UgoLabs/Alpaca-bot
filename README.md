# Alpaca Trading Bot 🦙

An advanced AI-powered trading system for Alpaca Markets, featuring Swing Trading, Day Trading, and Scalping bots powered by a GRU-Dual-DQN Reinforcement Learning agent.

## 🌟 Features
- **3 Specialized Bots**:
  - `Money Scraper`: High-frequency scalper (WebSocket) for quick gains.
  - `Swing Trader`: Multi-day position trader using ATR-based risk management.
  - `Day Trader`: Intraday scalper with tight stops.
- **Advanced AI Model**: Recurrent Dueling DQN (DRQN) with GRU layers to understand time-series patterns.
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
```

### 2. Train the Model
Train a fresh model on your watchlist:
```bash
python scripts/train.py --fresh --episodes 50
```

### 3. Run a Bot
Run any bot using the unified runner:
```bash
# Run Money Scraper
python scripts/run_bot.py money_scraper

# Run Swing Trader
python scripts/run_bot.py swing_trader

# Run Day Trader
python scripts/run_bot.py day_trader
```

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
The agent uses a **Recurrent Dueling DQN**:
1. **Input**: 20-candle window of price action + Technical Indicators (RSI, MACD, BB, etc.) + Portfolio State.
2. **GRU Layer**: Processes the time-series data to capture temporal dependencies.
3. **Dueling Heads**: Separates Value $V(s)$ and Advantage $A(s,a)$ streams.
4. **Action**: Buy, Sell, or Hold.

## ⚠️ Disclaimer
This software is for educational purposes only. Do not risk money you cannot afford to lose. Use at your own risk.