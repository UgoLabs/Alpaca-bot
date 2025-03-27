# Alpaca Trading Bot with Dueling DQN

This repository contains a trading bot that uses a Dueling Deep Q-Network (DQN) to learn trading strategies based on historical market data from yfinance. The bot is designed to trade on Alpaca markets using your account credentials.

## Features

- Dueling DQN architecture for better Q-value estimation
- Training on 5 years of historical stock data
- Integration with Alpaca API for account information
- GPU acceleration with TensorFlow
- Risk management with stop losses and trailing stops
- Configurable training parameters through .env file

## Requirements

- Python 3.8+
- NVIDIA GPU (RTX 4070 recommended)
- 64 GB RAM
- Alpaca trading account

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Configure your `.env` file with Alpaca API keys and trading parameters
4. Make sure your list of stocks is in `my_portfolio.txt` (one ticker per line)

## Usage

Run the training script:

```
python src/train_dueling_dqn.py
```

This will:
1. Load your portfolio stocks from `my_portfolio.txt`
2. Fetch 5 years of historical data for each stock
3. Train a Dueling DQN model for each stock
4. Save the best performing models to the `models/` directory

### Evaluating Models

After training, you can evaluate the models on recent data:

```
python src/evaluate_model.py --symbol AAPL
```

This will:
1. Load the best model for the specified stock
2. Evaluate it on 1 year of recent data
3. Generate a performance chart showing buy/sell actions and portfolio value
4. Compare the model's performance against a buy-and-hold strategy

Additional options:
- `--model`: Specify a custom model path
- `--period`: Change the evaluation period (default: 1y)

Example:
```
python src/evaluate_model.py --symbol NVDA --period 6mo
```

### Live Trading

To use the trained models for live trading on Alpaca:

```
python src/live_trading.py
```

By default, this will:
1. Load the best models for all stocks in your portfolio
2. Connect to your Alpaca account using API keys from .env
3. Execute trades based on model predictions
4. Set stop losses automatically for risk management
5. Run in a continuous loop, checking for new trading opportunities every 15 minutes

Additional options:
- `--symbols`: Specify a subset of symbols to trade (e.g., `--symbols AAPL MSFT GOOGL`)
- `--interval`: Set the trading check interval in minutes (default: 15)
- `--model-dir`: Specify a custom directory for model files

Example:
```
python src/live_trading.py --symbols AAPL MSFT --interval 30
```

## Configuration

All parameters can be configured in the `.env` file:

### Alpaca API Settings
- `ALPACA_API_KEY`: Your Alpaca API key
- `ALPACA_API_SECRET`: Your Alpaca API secret
- `ALPACA_API_BASE_URL`: Alpaca API base URL

### Training Settings
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Learning rate for the optimizer
- `GAMMA`: Discount factor for future rewards
- `HIDDEN_SIZE`: Size of hidden layers in the network

### Risk Management
- `MAX_POSITION_SIZE`: Maximum position size as fraction of portfolio
- `STOP_LOSS_PERCENTAGE`: Stop loss percentage
- `TRAILING_STOP_PERCENTAGE`: Trailing stop percentage

### GPU Settings
- `USE_GPU`: Enable/disable GPU usage
- `GPU_MEMORY_FRACTION`: Fraction of GPU memory to use
- `USE_MIXED_PRECISION`: Enable mixed precision training

## Model Architecture

The Dueling DQN architecture separates the state value and advantage functions, providing better learning efficiency especially for states where actions don't affect the environment significantly.

## Logs

Training logs are saved to:
- `trading_dqn.log`: Main log file
- `dueling_dqn_train.log`: Training script log

## License

This project is for personal use only. 