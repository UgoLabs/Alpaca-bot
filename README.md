# Alpaca Trading Bot

This repository contains a sophisticated trading bot that uses Alpaca Markets API for automated trading.

## Overview

The Alpaca Trading Bot is designed to automate trading strategies using machine learning and technical analysis. It can be configured for both paper trading and live trading through the Alpaca API.

## Features

- Real-time market data analysis
- Multiple trading strategies
- Machine learning prediction models
- Backtesting capabilities
- Portfolio optimization
- Sentiment analysis integration

## Requirements

See the requirements.txt file for a complete list of dependencies.

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your Alpaca API keys in the .env file
4. Run the bot: `python main.py`

## Configuration

The bot can be configured through various JSON configuration files:

- config.json: Main configuration settings
- trading_config.json: Trading strategy parameters
- oracle_config.json: Oracle deployment settings

## Components

### Data Fetchers

- alpaca_data_fetcher.py: Fetches market data from Alpaca API
- finnhub_data_fetcher.py: Fetches data from Finnhub API
- alpha_vantage_fetcher.py: Fetches data from Alpha Vantage API
- google_data_fetcher.py: Fetches data from Google APIs

### Trading Logic

- trading_strategies.py: Contains various trading strategies
- portfolio_optimizer.py: Optimizes portfolio allocation
- backtesting.py: Allows for strategy backtesting

### ML Components

- model_trainer.py: Trains ML models for prediction
- ml_predictor.py: Makes predictions using trained models
- sentiment_analyzer.py: Analyzes market sentiment

### Deployment

- run_bot.py: Main script to run the trading bot
- oracle_deploy.py: Deploys the oracle for automated trading
- check_status.py: Checks the status of running bots
- stop_trading_bot.py: Safely stops trading operations

## Utilities

- cleanup.py: Cleans up old data and logs
- fix_indentation.py: Fixes code indentation issues
- fix_parentheses.py: Fixes parentheses balancing issues

## Directories

- AlpacaBot/: Core bot implementation files
- DataLoader/: Data loading and processing modules
- DeepRLAgent/: Deep reinforcement learning agent
- EncoderDecoderAgent/: Encoder-decoder models
- models/: Model definitions
- trained_models/: Saved trained models
- data/: Data storage
- logs/: Log files
- Results/: Trading results and analysis
- PatternDetectionInCandleStick/: Candlestick pattern detection

## License

Proprietary - All rights reserved

## Disclaimer

This trading bot is provided for educational and research purposes only. Use at your own risk. The creators are not responsible for any financial losses incurred through the use of this software.