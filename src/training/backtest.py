"""
Backtesting Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Backtester:
    def __init__(self, agent, env_class):
        self.agent = agent
        self.env_class = env_class
        
    def run(self, df, symbol="Unknown"):
        env = self.env_class(df)
        state = env.reset()
        done = False
        
        history = {
            'dates': df.index[env.window_size + 200:],
            'price': [],
            'net_worth': [],
            'actions': [],
        }
        
        while not done:
            action = self.agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            
            history['price'].append(env._get_price())
            history['net_worth'].append(env.net_worth)
            history['actions'].append(action)
            
            state = next_state
        
        metrics = env.get_metrics()
        return metrics, history
