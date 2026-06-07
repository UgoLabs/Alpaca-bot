# Alpaca Trading Bot

Swing equity and options spread bots powered by a multi-modal ensemble agent (time series + news).

## Bots

| Mode | Command | Config |
|------|---------|--------|
| Swing (live) | `python manage.py run swing` | `SwingTraderConfig` |
| Paper swing | `python manage.py run paper_swing` | Same as swing, paper keys |
| Options spreads | `python manage.py run options` | `OptionsTraderConfig` (paper only) |

## Project layout

```
alpaca-bot/
├── config/          # settings.py, watchlists
├── src/
│   ├── bots/        # multimodal_trader.py (live loop)
│   ├── agents/      # ensemble agent
│   ├── data/        # pipeline, market/options data
│   └── execution/   # options spread broker
├── scripts/         # train, backtest, download, ops
└── models/          # checkpoint .pth files
```

## Setup

1. Copy `.env.example` → `.env` and add Alpaca API keys.
2. Build swing CSVs: `python scripts/download_data.py`
3. For options: `python scripts/download_options_bars.py --watchlist config/watchlists/options_liquid_200.txt`

## Train

```bash
python manage.py train
python manage.py status
```

## Backtest

```bash
python scripts/backtest_swing_portfolio.py
python scripts/backtest_options_portfolio.py
```

## Docker

```bash
docker compose up --build -d swing_bot
docker compose up --build -d options_bot
docker compose up --build -d paper_swing_bot
```
