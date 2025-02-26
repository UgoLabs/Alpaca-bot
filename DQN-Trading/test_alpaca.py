from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import os

def test_alpaca():
    load_dotenv()
    try:
        # Initialize Alpaca API (Paper Trading)
        api = tradeapi.REST(
            key_id=os.getenv('PAPER_API_KEY'),
            secret_key=os.getenv('PAPER_API_SECRET_KEY'),
            base_url=os.getenv('PAPER_BASE_URL')
        )
        
        # Test connection
        account = api.get_account()
        print(f"Connected to Alpaca successfully!")
        print(f"Account Status: {account.status}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        return True
    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return False

if __name__ == "__main__":
    test_alpaca()