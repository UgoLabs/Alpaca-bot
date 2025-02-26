import os
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

paper_api = tradeapi.REST(
    key_id=os.getenv('PAPER_API_KEY'),
    secret_key=os.getenv('PAPER_API_SECRET_KEY'),
    base_url=os.getenv('PAPER_BASE_URL', 'https://paper-api.alpaca.markets')
)

live_api = tradeapi.REST(
    key_id=os.getenv('APCA_API_KEY'),
    secret_key=os.getenv('APCA_API_SECRET_KEY'),
    base_url=os.getenv('APCA_API_BASE_URL', 'https://api.alpaca.markets')
)

def make_investment(self, action_list):
    """
    Given a list of actions, update the data-frame with the corresponding trade actions.
    """
    self.data.loc[:, self.action_name] = 'None'
    i = self.start_index_reward + 1
    for a in action_list:
        self.data.loc[self.data.index[i], self.action_name] = self.code_to_action[a]
        i += 1
