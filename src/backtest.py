"""
Backtesting Script for Swing Trading DQN
Tests the trained model on historical data and generates performance metrics.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from swing_model import DuelingDQN
from utils import add_technical_indicators, normalize_state, get_state_size, detect_market_regime

# =============================================================================
# Configuration
# =============================================================================
WINDOW_SIZE = 20
INITIAL_BALANCE = 10000
COMMISSION = 0.0  # Alpaca is commission-free
MAX_POSITION_PCT = 0.25  # Max 25% of portfolio per position
RISK_PER_TRADE = 0.02  # Risk 2% per trade
ATR_STOP_MULTIPLIER = 2.5  # Match environment settings
ATR_TRAILING_MULTIPLIER = 3.0

# =============================================================================
# Backtester Class
# =============================================================================
class Backtester:
    def __init__(self, model_path, initial_balance=INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.model_path = model_path
        
        # Load model - use dynamic state size from utils
        state_size = get_state_size(WINDOW_SIZE)
        action_size = 3
        # Must match training configuration (Noisy Nets = True)
        self.agent = DuelingDQN(state_size, action_size, use_noisy=True)
        
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def run_backtest(self, symbol, start_date, end_date):
        """Run backtest on a single symbol."""
        print(f"\n{'='*60}")
        print(f"Backtesting {symbol} from {start_date} to {end_date}")
        print(f"{'='*60}")
        
        # Fetch data
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if len(df) < WINDOW_SIZE + 50:
            print(f"Not enough data for {symbol}")
            return None
        
        # Add indicators
        df = add_technical_indicators(df)
        
        # Initialize tracking
        balance = self.initial_balance
        shares = 0
        entry_price = 0
        highest_price = 0
        
        # Performance tracking
        portfolio_values = []
        trades = []
        actions_taken = []
        
        # Buy and hold benchmark
        buy_hold_shares = self.initial_balance / df['Close'].iloc[WINDOW_SIZE]
        
        # Run through each day
        for step in range(WINDOW_SIZE, len(df)):
            current_price = df['Close'].iloc[step]
            current_atr = df['atr'].iloc[step]
            date = df.index[step]
            
            # Get state
            market_state = normalize_state(df, step, WINDOW_SIZE)
            
            # Regime features are already included in normalize_state
            # So we remove the manual addition here to avoid double counting and size mismatch
            pass
            
            # Portfolio state
            unrealized_pnl = 0
            if shares > 0:
                unrealized_pnl = (current_price - entry_price) / entry_price
            
            portfolio_state = np.array([
                balance / self.initial_balance,
                (shares * current_price) / self.initial_balance,
                unrealized_pnl,
                1.0 if shares > 0 else 0.0,
                0.0 # 5th feature (likely held_time or similar), added to match model size 231
            ])
            
            state = np.concatenate((market_state, portfolio_state))
            
            # Get action from model
            action = self.agent.act(state, epsilon=0.0)
            original_action = action
            
            # Risk Management: Check Stops
            stop_triggered = False
            if shares > 0:
                # Update trailing high
                if current_price > highest_price:
                    highest_price = current_price
                
                # Hard stop loss
                stop_price = entry_price - (current_atr * ATR_STOP_MULTIPLIER)
                if current_price < stop_price:
                    action = 2
                    stop_triggered = True
                
                # Trailing stop
                trailing_stop = highest_price - (current_atr * ATR_TRAILING_MULTIPLIER)
                if current_price < trailing_stop:
                    action = 2
                    stop_triggered = True
            
            # Execute action
            if action == 1 and shares == 0:  # BUY
                # Position sizing based on ATR
                risk_amount = balance * RISK_PER_TRADE
                stop_distance = current_atr * ATR_STOP_MULTIPLIER
                position_size = risk_amount / stop_distance
                
                # Cap at max position
                max_shares = (balance * MAX_POSITION_PCT) / current_price
                shares_to_buy = min(int(position_size), int(max_shares))
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    balance -= cost
                    shares = shares_to_buy
                    entry_price = current_price
                    highest_price = current_price
                    
                    trades.append({
                        'date': str(date),
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'balance': balance
                    })
            
            elif action == 2 and shares > 0:  # SELL
                revenue = shares * current_price
                profit = revenue - (shares * entry_price)
                profit_pct = profit / (shares * entry_price) * 100
                
                balance += revenue
                
                trades.append({
                    'date': str(date),
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'stop_triggered': stop_triggered,
                    'balance': balance
                })
                
                shares = 0
                entry_price = 0
                highest_price = 0
            
            # Track portfolio value
            portfolio_value = balance + (shares * current_price)
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'buy_hold_value': buy_hold_shares * current_price,
                'action': ['HOLD', 'BUY', 'SELL'][original_action],
                'price': current_price
            })
            
            actions_taken.append(action)
        
        # Final metrics
        final_value = portfolio_values[-1]['portfolio_value']
        buy_hold_final = portfolio_values[-1]['buy_hold_value']
        
        results = self._calculate_metrics(portfolio_values, trades, final_value, buy_hold_final)
        results['symbol'] = symbol
        results['trades'] = trades
        results['portfolio_values'] = portfolio_values
        
        return results
    
    def _calculate_metrics(self, portfolio_values, trades, final_value, buy_hold_final):
        """Calculate performance metrics."""
        pv_df = pd.DataFrame(portfolio_values)
        pv_df['returns'] = pv_df['portfolio_value'].pct_change()
        pv_df['buy_hold_returns'] = pv_df['buy_hold_value'].pct_change()
        
        # Total Return
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        buy_hold_return = (buy_hold_final - self.initial_balance) / self.initial_balance * 100
        
        # Win Rate
        winning_trades = [t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t['action'] == 'SELL' and t.get('profit', 0) <= 0]
        total_sells = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_sells * 100 if total_sells > 0 else 0
        
        # Average Win/Loss
        avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Profit Factor
        total_wins = sum([t['profit'] for t in winning_trades]) if winning_trades else 0
        total_losses = abs(sum([t['profit'] for t in losing_trades])) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe Ratio (annualized)
        risk_free_rate = 0.05 / 252  # ~5% annual
        excess_returns = pv_df['returns'].dropna() - risk_free_rate
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Max Drawdown
        cummax = pv_df['portfolio_value'].cummax()
        drawdown = (pv_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Sortino Ratio
        negative_returns = pv_df['returns'][pv_df['returns'] < 0]
        downside_std = negative_returns.std()
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return_pct': total_return,
            'buy_hold_return_pct': buy_hold_return,
            'alpha': total_return - buy_hold_return,
            'total_trades': total_sells,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_drawdown
        }
    
    def print_results(self, results):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {results['symbol']}")
        print(f"{'='*60}")
        print(f"Initial Balance:     ${results['initial_balance']:,.2f}")
        print(f"Final Value:         ${results['final_value']:,.2f}")
        print(f"Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"Buy & Hold Return:   {results['buy_hold_return_pct']:+.2f}%")
        print(f"Alpha (vs B&H):      {results['alpha']:+.2f}%")
        print(f"-"*60)
        print(f"Total Trades:        {results['total_trades']}")
        print(f"Win Rate:            {results['win_rate_pct']:.1f}%")
        print(f"Avg Win:             {results['avg_win_pct']:+.2f}%")
        print(f"Avg Loss:            {results['avg_loss_pct']:.2f}%")
        print(f"Profit Factor:       {results['profit_factor']:.2f}")
        print(f"-"*60)
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"{'='*60}")
    
    def plot_results(self, results, save_path=None):
        """Plot backtest results."""
        pv_df = pd.DataFrame(results['portfolio_values'])
        pv_df['date'] = pd.to_datetime(pv_df['date'])
        pv_df.set_index('date', inplace=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. Portfolio Value vs Buy & Hold
        ax1 = axes[0]
        ax1.plot(pv_df.index, pv_df['portfolio_value'], label='DQN Strategy', linewidth=2)
        ax1.plot(pv_df.index, pv_df['buy_hold_value'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title(f"{results['symbol']} - Portfolio Value")
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark buy/sell points
        buys = pv_df[pv_df['action'] == 'BUY']
        sells = pv_df[pv_df['action'] == 'SELL']
        ax1.scatter(buys.index, buys['portfolio_value'], marker='^', color='green', s=100, label='Buy', zorder=5)
        ax1.scatter(sells.index, sells['portfolio_value'], marker='v', color='red', s=100, label='Sell', zorder=5)
        
        # 2. Drawdown
        ax2 = axes[1]
        cummax = pv_df['portfolio_value'].cummax()
        drawdown = (pv_df['portfolio_value'] - cummax) / cummax * 100
        ax2.fill_between(pv_df.index, drawdown, 0, alpha=0.5, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Price with Actions
        ax3 = axes[2]
        ax3.plot(pv_df.index, pv_df['price'], label='Price', linewidth=1, color='gray')
        ax3.scatter(buys.index, buys['price'], marker='^', color='green', s=100, label='Buy')
        ax3.scatter(sells.index, sells['price'], marker='v', color='red', s=100, label='Sell')
        ax3.set_title('Price Action with Trades')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Chart saved to {save_path}")
        
        plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest DQN Trading Model')
    parser.add_argument('--symbol', type=str, default='SPY', help='Stock symbol')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-01', help='End date')
    parser.add_argument('--model', type=str, default=None, help='Model path (default: models/SYMBOL_dqn_best.pth)')
    parser.add_argument('--plot', action='store_true', help='Show plot')
    
    args = parser.parse_args()
    
    # Default model path if not provided
    if args.model is None:
        args.model = f"models/{args.symbol}_dqn_best.pth"
    
    # Run backtest
    try:
        backtester = Backtester(args.model, initial_balance=INITIAL_BALANCE)
        results = backtester.run_backtest(args.symbol, args.start, args.end)
        
        if results:
            backtester.print_results(results)
            
            if args.plot:
                backtester.plot_results(results, save_path=f"logs/{args.symbol}_backtest.png")
            
            # Save results
            os.makedirs("logs", exist_ok=True)
            results_copy = {k: v for k, v in results.items() if k not in ['trades', 'portfolio_values']}
            with open(f"logs/{args.symbol}_backtest_results.json", 'w') as f:
                json.dump(results_copy, f, indent=2)
            print(f"\nResults saved to logs/{args.symbol}_backtest_results.json")
    except Exception as e:
        print(f"Error running backtest: {e}")

if __name__ == "__main__":
    main()
