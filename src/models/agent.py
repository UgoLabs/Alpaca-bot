"""
DQN Agent Wrapper
Handles model training, action selection, and persistence
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os

from .network import DuelingNetwork
from config.settings import TrainingConfig


class DuelingDQN:
    """
    Dueling DQN Agent with Double DQN and optional Noisy Networks.
    """
    
    def __init__(self, state_size, action_size, learning_rate=None, use_noisy=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_noisy = use_noisy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if learning_rate is None:
            learning_rate = TrainingConfig.LEARNING_RATE
        
        # Networks
        self.model = DuelingNetwork(state_size, action_size, use_noisy=use_noisy).to(self.device)
        self.target_model = DuelingNetwork(state_size, action_size, use_noisy=use_noisy).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.HuberLoss()
    
    def act(self, state, epsilon=0.0):
        """Select action using epsilon-greedy (or pure greedy if noisy)."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Note: We do NOT switch to eval() here to avoid race conditions 
        # with backward() in parallel training threads.
        # Dropout being active adds noise/exploration which is acceptable.
        
        with torch.no_grad():
            q_values = self.model(state)
        
        if self.use_noisy:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            if np.random.rand() <= epsilon:
                return random.randrange(self.action_size)
            return np.argmax(q_values.cpu().data.numpy())
    
    def train_step(self, states, targets, weights=None):
        """Single training step on a batch."""
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_noisy:
            self.model.reset_noise()
            self.target_model.reset_noise()
        
        predictions = self.model(states)
        
        if weights is not None:
            loss = torch.mean(weights * self.loss_fn(predictions, targets))
        else:
            loss = self.loss_fn(predictions, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_model(self):
        """Hard update target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def soft_update_target_model(self, tau=0.005):
        """Soft update target network."""
        for target_param, local_param in zip(self.target_model.parameters(), 
                                              self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filepath):
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights with architecture mismatch handling."""
        if os.path.exists(filepath):
            try:
                state_dict = torch.load(filepath, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Model loaded from {filepath}")
            except Exception as e:
                print(f"Warning: Could not load model ({e}). Starting fresh.")
        else:
            print(f"Model file not found: {filepath}. Starting fresh.")
