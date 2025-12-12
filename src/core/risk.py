"""
Risk Management Functions
Centralized position sizing and risk limit checks
"""
import numpy as np
from config.settings import RISK_PER_TRADE, MAX_POSITION_PCT


def calculate_position_size(equity, entry_price, stop_distance, 
                            risk_per_trade=None, max_position_pct=None):
    """
    Calculate position size based on risk parameters.
    
    Args:
        equity: Account equity in dollars
        entry_price: Planned entry price
        stop_distance: Distance to stop loss in dollars
        risk_per_trade: Fraction of equity to risk (default from config)
        max_position_pct: Max fraction of equity per position (default from config)
    
    Returns:
        int: Number of shares to buy (floored)
    """
    if risk_per_trade is None:
        risk_per_trade = RISK_PER_TRADE
    if max_position_pct is None:
        max_position_pct = MAX_POSITION_PCT
    
    if stop_distance <= 0 or entry_price <= 0:
        return 0
    
    # Risk-based sizing: shares = risk_amount / stop_distance
    risk_amount = equity * risk_per_trade
    shares_from_risk = risk_amount / stop_distance
    
    # Cap at max position percentage
    max_capital = equity * max_position_pct
    shares_from_cap = max_capital / entry_price
    
    # Return the smaller of the two, floored to integer
    return int(min(shares_from_risk, shares_from_cap))


def calculate_atr_stops(entry_price, atr, 
                        stop_mult=2.5, trailing_mult=3.0, profit_mult=4.0):
    """
    Calculate ATR-based stop loss, trailing stop, and take profit levels.
    
    Args:
        entry_price: Entry price of position
        atr: Current ATR value
        stop_mult: Multiplier for hard stop loss
        trailing_mult: Multiplier for trailing stop activation
        profit_mult: Multiplier for take profit
    
    Returns:
        dict with stop_loss, trailing_trigger, take_profit prices
    """
    return {
        'stop_loss': entry_price - (atr * stop_mult),
        'trailing_trigger': entry_price + (atr * trailing_mult),
        'take_profit': entry_price + (atr * profit_mult)
    }


def check_risk_limits(account, positions, max_positions=None, max_portfolio_risk=0.20):
    """
    Check if we can take on more risk.
    
    Args:
        account: Alpaca account object
        positions: List of current positions
        max_positions: Maximum number of positions allowed
        max_portfolio_risk: Maximum fraction of equity at risk
    
    Returns:
        dict with 'can_trade', 'reason', 'available_slots'
    """
    equity = float(account.equity)
    num_positions = len(positions)
    
    # Check position limit
    if max_positions and num_positions >= max_positions:
        return {
            'can_trade': False,
            'reason': f'Max positions ({max_positions}) reached',
            'available_slots': 0
        }
    
    # Check buying power
    buying_power = float(account.buying_power)
    if buying_power < equity * 0.01:  # Less than 1% buying power
        return {
            'can_trade': False,
            'reason': 'Insufficient buying power',
            'available_slots': 0
        }
    
    # Check total risk exposure
    total_risk = sum(
        abs(float(p.unrealized_pl)) for p in positions
    )
    if total_risk > equity * max_portfolio_risk:
        return {
            'can_trade': False,
            'reason': f'Portfolio risk exceeds {max_portfolio_risk*100}%',
            'available_slots': 0
        }
    
    return {
        'can_trade': True,
        'reason': 'OK',
        'available_slots': (max_positions - num_positions) if max_positions else 999
    }


def calculate_pnl_pct(entry_price, current_price):
    """Calculate percentage P&L"""
    if entry_price <= 0:
        return 0.0
    return (current_price - entry_price) / entry_price
