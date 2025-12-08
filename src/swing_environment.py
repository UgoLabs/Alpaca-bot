"""
Swing Trading Environment for Dueling DQN
Implements proper risk management, position sizing, and market regime awareness.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from utils import add_technical_indicators, normalize_state, get_state_size, detect_market_regime


class SwingTradingEnv(gym.Env):
    """
    A professional-grade Gym environment for Swing Trading.
    
    Features:
    - ATR-based position sizing (risk 1-2% per trade)
    - Market regime filtering (only trade with trend)
    - Trailing stops and hard stops
    - Anti-churn logic (minimum holding period)
    - Cooldown after losses
    - Risk-adjusted reward function
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.0, 
                 window_size=20, risk_per_trade=0.02, max_position_pct=0.25,
                 skip_indicators=False):
        super(SwingTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.risk_per_trade = risk_per_trade      # Risk 2% per trade
        self.max_position_pct = max_position_pct  # Max 25% in single position

        # Add indicators only if not already computed
        if not skip_indicators:
            self.df = add_technical_indicators(self.df)
        
        # Define Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Define Observation Space (using helper function)
        self.state_size = get_state_size(window_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_size,), 
            dtype=np.float32
        )

        # =====================================================================
        # Risk Management Parameters
        # =====================================================================
        self.stop_loss_atr_multiplier = 2.5      # Wider stop for swing trading
        self.trailing_stop_atr_multiplier = 3.0  # Trail at 3x ATR
        self.min_holding_period = 3              # Minimum 3 days to prevent churn
        self.cooldown_steps = 5                  # Days to wait after a loss
        self.max_consecutive_losses = 3          # Reduce size after 3 losses
        
        # NEW: Take Profit Settings
        self.take_profit_atr_multiplier = 4.0    # Take profit at 4x ATR gain
        self.partial_take_profit_enabled = True  # Partial profits at 2x ATR
        
        # NEW: Volume Confirmation
        self.require_volume_confirmation = True  # Only buy with volume
        self.volume_threshold = 1.5              # 50% above average volume
        
        # NEW: RSI Filters
        self.rsi_overbought = 70                 # Don't buy above this
        self.rsi_oversold = 30                   # Consider exit below this if short
        self.rsi_extreme_overbought = 80         # Strong sell signal
        
        # NEW: Pullback Entry Detection
        self.pullback_threshold = 0.3            # Buy when BB %B < 0.3 in uptrend
        
        # Market regime settings
        self.require_trend_alignment = True      # Only trade with trend
        self.min_adx_for_trend = 20              # ADX threshold for trend
        
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = self.window_size + 200  # Skip warm-up period for indicators

        # Position tracking
        self.entry_price = 0
        self.entry_step = 0
        self.highest_price_since_entry = 0
        self.position_atr_at_entry = 0
        
        # Performance tracking
        self.trades = []
        self.returns = []
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Risk management state
        self.cooldown_counter = 0
        self.in_drawdown = False
        
        return self._next_observation()

    def _next_observation(self):
        """Get the current state observation."""
        # Get market features from utils
        market_state = normalize_state(self.df, self.current_step, self.window_size)
        
        # Get current values
        current_price = self._get_price()
        current_atr = self.df['atr'].iloc[self.current_step]
        
        # Portfolio state (5 features)
        unrealized_pnl = 0
        days_in_position = 0
        if self.shares > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            days_in_position = self.current_step - self.entry_step
            
        portfolio_state = np.array([
            self.balance / self.initial_balance,                    # Normalized balance
            (self.shares * current_price) / self.initial_balance,   # Position size
            unrealized_pnl,                                          # Current P/L %
            min(days_in_position / 20, 1.0),                        # Holding period (normalized)
            1.0 if self.shares > 0 else 0.0                         # In position flag
        ], dtype=np.float32)
        
        return np.concatenate((market_state, portfolio_state))

    def _get_price(self):
        """Safely get current price."""
        price = self.df['Close'].iloc[self.current_step]
        if hasattr(price, 'iloc'):
            return float(price.iloc[0])
        return float(price)
    
    def _get_indicator(self, name):
        """Safely get indicator value."""
        val = self.df[name].iloc[self.current_step]
        if hasattr(val, 'iloc'):
            return float(val.iloc[0])
        return float(val)

    def _calculate_position_size(self, current_price, current_atr):
        """
        Calculate position size based on ATR and risk parameters.
        Risk a fixed percentage of portfolio per trade.
        """
        # Risk amount (e.g., 2% of current portfolio)
        risk_amount = self.net_worth * self.risk_per_trade
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            risk_amount *= 0.5  # Half size after 3 consecutive losses
        
        # Stop distance based on ATR
        stop_distance = current_atr * self.stop_loss_atr_multiplier
        
        # Position size based on risk
        if stop_distance > 0:
            risk_based_shares = risk_amount / stop_distance
        else:
            risk_based_shares = 0
        
        # Max position cap (e.g., 25% of portfolio)
        max_position_value = self.net_worth * self.max_position_pct
        max_affordable = self.balance / current_price
        max_shares = min(max_position_value / current_price, max_affordable)
        
        # Take the smaller of risk-based or max cap
        shares = int(min(risk_based_shares, max_shares))
        
        return max(shares, 0)

    def _check_market_regime(self):
        """
        Check if current market regime is favorable for trading.
        Returns: (can_buy, can_sell, regime_info)
        """
        # Get necessary indicators
        current_price = self._get_price()
        sma_50 = self._get_indicator('sma_50')
        sma_200 = self._get_indicator('sma_200')
        adx = self._get_indicator('adx')
        bb_width = self._get_indicator('bb_width')
        
        can_buy = True
        can_sell = True
        
        # 1. Choppy Market Filter (Sideways Death Trap)
        # Avoid buying if ADX is low (weak trend)
        is_choppy = adx < 20
        
        # 2. Fast Trend Filter
        # Buy if price is above SMA 50 (Faster than waiting for Golden Cross)
        is_uptrend = current_price > sma_50
        
        # 3. Major Trend Filter (Background)
        # Ideally price should also be above SMA 200, or SMA 50 rising
        major_trend = current_price > sma_200
        
        if self.require_trend_alignment:
            # Block buys in choppy markets or downtrends
            if is_choppy or not is_uptrend:
                can_buy = False
            
            # Stricter: Block if major trend is down, unless it's a strong reversal (not implemented yet)
            if not major_trend:
                # Allow if short-term momentum is very strong (e.g. Price > SMA20 not implemented here)
                # For now, stick to safer logic
                can_buy = False

        regime_info = {
            'regime': 1 if is_uptrend and major_trend else (-1 if not major_trend else 0),
            'trend_strength': adx / 100.0,
            'adx': adx,
            'is_choppy': is_choppy
        }
        
        return can_buy, can_sell, regime_info

    def _check_stops(self, current_price, current_atr):
        """
        Check if stop loss, trailing stop, or take profit should be triggered.
        Returns: (should_sell, stop_type)
        """
        if self.shares == 0:
            return False, None
        
        # Update highest price for trailing stop
        if current_price > self.highest_price_since_entry:
            self.highest_price_since_entry = current_price
        
        # Use ATR at entry for stop calculations (more stable)
        atr_for_stops = self.position_atr_at_entry if self.position_atr_at_entry > 0 else current_atr
        
        # 1. Hard Stop Loss (2.5x ATR below entry)
        hard_stop = self.entry_price - (atr_for_stops * self.stop_loss_atr_multiplier)
        if current_price < hard_stop:
            return True, 'hard_stop'
        
        # 2. NEW: Dynamic Take Profit Target
        # If trend is strong (ADX > 30), aim higher (6x ATR)
        # If trend is weak/normal, aim for standard (4x ATR)
        adx = self._get_indicator('adx')
        dynamic_multiplier = 6.0 if adx > 30 else self.take_profit_atr_multiplier
        
        take_profit = self.entry_price + (atr_for_stops * dynamic_multiplier)
        if current_price >= take_profit:
            return True, 'take_profit'
        
        # 3. NEW: RSI Extreme Exit (RSI > 80 = very overbought)
        rsi = self._get_indicator('rsi')
        if rsi > self.rsi_extreme_overbought and current_price > self.entry_price:
            return True, 'rsi_extreme'
        
        # 4. Trailing Stop (3x ATR below highest price)
        # Only activate trailing stop if we're in profit
        if current_price > self.entry_price:
            trailing_stop = self.highest_price_since_entry - (atr_for_stops * self.trailing_stop_atr_multiplier)
            if current_price < trailing_stop:
                return True, 'trailing_stop'
        
        # 5. Time-based stop (optional): Exit if no progress after 20 days
        days_held = self.current_step - self.entry_step
        if days_held > 20 and current_price < self.entry_price:
            return True, 'time_stop'
        
        return False, None

    def _calculate_reward(self, action, profit_pct, stop_type, regime_info, is_pullback_entry=False):
        """
        Calculate risk-adjusted reward.
        Rewards good decisions, penalizes bad ones.
        """
        reward = 0
        
        if action == 2 and self.shares > 0:  # Completed a trade
            # Base reward is profit percentage
            reward = profit_pct * 100
            
            # Bonus for winning trades
            if profit_pct > 0:
                reward += 2.0  # Win bonus
                # Extra bonus for riding winners
                if profit_pct > 0.05:  # >5% gain
                    reward += 2.0
                if profit_pct > 0.10:  # >10% gain
                    reward += 3.0
                
                # NEW: Extra bonus for taking profit at target
                if stop_type == 'take_profit':
                    reward += 2.0  # Reward for hitting target
                
                # NEW: Bonus for RSI extreme exit (selling at top)
                if stop_type == 'rsi_extreme':
                    reward += 1.5  # Good timing bonus
            else:
                # Smaller penalty for stopped out (risk was managed)
                if stop_type in ['hard_stop', 'trailing_stop']:
                    reward -= 0.5  # Controlled loss
                else:
                    reward -= 1.5  # Manual sell at loss
            
            # Penalize overtrading
            days_held = self.current_step - self.entry_step
            if days_held < self.min_holding_period:
                reward -= 2.0  # Penalty for churning
        
        elif action == 1:  # Bought
            # Small penalty for buying in bad regime
            if regime_info['regime'] == -1:
                reward -= 1.0
            # Small bonus for buying in good regime
            elif regime_info['regime'] == 1 and regime_info['adx'] > 25:
                reward += 0.5
            
            # NEW: Extra bonus for pullback entries (buying dips in uptrends)
            # CRITICAL FIX: Only award this bonus if the trade ends up being a WINNER
            # We can't know this at buy time easily in this structure without looking ahead or storing state.
            # However, since this reward function is called at every step, we can give a *small* immediate reward
            # but the REAL bonus should be conditioned on profit. 
            # Current implementation mainly rewards *holding* or *selling*.
            # The 'action == 1' block handles the immediate reward for taking the action.
            
            # Adjusted: Small immediate reward for good process, but reduced to prevent hacking.
            if is_pullback_entry:
                reward += 0.2  # Reduced from 1.0 to prevent farming losing trades
        
        elif action == 0:  # Held
            # Small reward for holding winners
            if self.shares > 0:
                current_pnl = (self._get_price() - self.entry_price) / self.entry_price
                if current_pnl > 0:
                    reward += 0.1  # Encourage holding winners
                    
        # Drawdown penalty
        if self.net_worth < self.max_net_worth * 0.9:  # >10% drawdown
            reward -= 0.5
            self.in_drawdown = True
        else:
            self.in_drawdown = False
        
        return reward

    def step(self, action):
        """Execute one step in the environment."""
        current_price = self._get_price()
        current_atr = self._get_indicator('atr')
        
        reward = 0
        done = False
        profit_pct = 0
        stop_type = None
        executed_action = action
        
        # Get market regime
        can_buy, can_sell, regime_info = self._check_market_regime()
        
        # Get additional indicators for new filters
        rsi = self._get_indicator('rsi')
        volume_ratio = self._get_indicator('volume_ratio')
        bb_pband = self._get_indicator('bb_pband')
        
        # =====================================================================
        # COOLDOWN CHECK
        # =====================================================================
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if action == 1:  # Block buys during cooldown
                executed_action = 0
        
        # =====================================================================
        # STOP LOSS CHECK (Takes priority)
        # =====================================================================
        should_stop, stop_type = self._check_stops(current_price, current_atr)
        if should_stop:
            executed_action = 2  # Force sell
        
        # =====================================================================
        # MINIMUM HOLDING PERIOD CHECK
        # =====================================================================
        if action == 2 and self.shares > 0:
            days_held = self.current_step - self.entry_step
            if days_held < self.min_holding_period and not should_stop:
                executed_action = 0  # Force hold (unless stopped out)
        
        # =====================================================================
        # MARKET REGIME CHECK
        # =====================================================================
        if action == 1 and not can_buy:
            executed_action = 0  # Block buy in bad regime
        
        # =====================================================================
        # NEW: VOLUME CONFIRMATION CHECK
        # =====================================================================
        if action == 1 and self.require_volume_confirmation:
            if volume_ratio < self.volume_threshold:
                executed_action = 0  # Block low-volume buys
        
        # =====================================================================
        # NEW: RSI OVERBOUGHT CHECK
        # =====================================================================
        if action == 1 and rsi > self.rsi_overbought:
            executed_action = 0  # Don't buy when overbought
        
        # =====================================================================
        # NEW: Track pullback entry quality for reward
        # =====================================================================
        is_pullback_entry = (regime_info['regime'] == 1 and bb_pband < self.pullback_threshold)
        
        # =====================================================================
        # EXECUTE ACTION
        # =====================================================================
        if executed_action == 1:  # Buy
            if self.shares == 0:  # Only if not already in position
                shares_to_buy = self._calculate_position_size(current_price, current_atr)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission_cost = cost * self.commission
                    
                    self.balance -= (cost + commission_cost)
                    self.shares = shares_to_buy
                    self.entry_price = current_price
                    self.entry_step = self.current_step
                    self.highest_price_since_entry = current_price
                    self.position_atr_at_entry = current_atr
                    
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'regime': regime_info['regime'],
                        'is_pullback_entry': is_pullback_entry,
                        'rsi': rsi,
                        'volume_ratio': volume_ratio
                    })
        
        elif executed_action == 2:  # Sell
            if self.shares > 0:
                revenue = self.shares * current_price
                commission_cost = revenue * self.commission
                
                # Calculate P/L
                profit = revenue - commission_cost - (self.shares * self.entry_price)
                profit_pct = profit / (self.shares * self.entry_price)
                
                self.balance += (revenue - commission_cost)
                self.returns.append(profit_pct)
                self.total_trades += 1
                
                # Track wins/losses
                if profit > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.cooldown_counter = self.cooldown_steps
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.shares,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'stop_type': stop_type,
                    'days_held': self.current_step - self.entry_step
                })
                
                # Reset position
                self.shares = 0
                self.entry_price = 0
                self.entry_step = 0
                self.highest_price_since_entry = 0
                self.position_atr_at_entry = 0

        # =====================================================================
        # UPDATE STATE
        # =====================================================================
        self.net_worth = self.balance + (self.shares * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # Calculate reward
        reward = self._calculate_reward(executed_action, profit_pct, stop_type, regime_info, is_pullback_entry)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            done = True
            # Final liquidation
            if self.shares > 0:
                self.balance += self.shares * self._get_price()
                self.shares = 0
                self.net_worth = self.balance
        
        # Get next observation
        obs = self._next_observation()
        
        # Info dict
        info = {
            'net_worth': self.net_worth,
            'action': executed_action,
            'stop_type': stop_type,
            'regime': regime_info['regime'],
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'consecutive_losses': self.consecutive_losses
        }
        
        return obs, reward, done, info

    def render(self, mode='human'):
        """Render the environment state."""
        win_rate = self.winning_trades / max(self.total_trades, 1) * 100
        print(f"Step: {self.current_step} | Net Worth: ${self.net_worth:,.2f} | "
              f"Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")

    def get_metrics(self):
        """Get performance metrics."""
        if len(self.returns) == 0:
            return {}
        
        returns = np.array(self.returns)
        
        return {
            'total_return': (self.net_worth - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'avg_return': np.mean(returns),
            'max_drawdown': (self.max_net_worth - self.net_worth) / self.max_net_worth,
            'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        }
