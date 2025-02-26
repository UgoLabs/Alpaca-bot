import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca Paper Trading Config
PAPER_TRADING_CONFIG = {
    'API_KEY': os.getenv('PAPER_API_KEY'),
    'API_SECRET': os.getenv('PAPER_API_SECRET_KEY'),
    'BASE_URL': os.getenv('PAPER_BASE_URL')
}

# Alpaca Live Trading Config
LIVE_TRADING_CONFIG = {
    'API_KEY': os.getenv('APCA_API_KEY'),
    'API_SECRET': os.getenv('APCA_API_SECRET_KEY'),
    'BASE_URL': os.getenv('APCA_API_BASE_URL')
}

# Other API Keys
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')