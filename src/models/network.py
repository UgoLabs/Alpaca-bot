"""
GRU-Enhanced Dueling DQN Network
Processes time-series via GRU, then dueling heads for value/advantage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config.settings import TrainingConfig


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration."""
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            return F.linear(x, 
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DuelingNetwork(nn.Module):
    """
    Recurrent Dueling DQN with GRU for time-series processing.
    
    Architecture:
    1. Split input into time-series (window) and static (portfolio) features
    2. Process time-series through GRU
    3. Concatenate GRU output with static features
    4. Dueling heads: Value stream + Advantage stream
    """
    
    def __init__(self, state_size, action_size, use_noisy=True):
        super(DuelingNetwork, self).__init__()
        self.use_noisy = use_noisy
        
        # Time-series structure (must match utils/state.py)
        self.window_size = TrainingConfig.WINDOW_SIZE
        self.num_window_features = TrainingConfig.NUM_WINDOW_FEATURES
        self.time_series_len = self.window_size * self.num_window_features
        self.static_features_len = state_size - self.time_series_len
        
        if self.static_features_len < 0:
            # Fallback to dense network if dimensions don't match
            self.use_gru = False
            print(f"Warning: State size {state_size} incompatible with GRU. Using Dense.")
            self.fc1 = nn.Linear(state_size, 256)
        else:
            self.use_gru = True
            
            # GRU for time-series
            self.gru_hidden_size = 128
            self.gru = nn.GRU(
                input_size=self.num_window_features,
                hidden_size=self.gru_hidden_size,
                num_layers=1,
                batch_first=True
            )
            
            # First FC layer takes GRU output + static features
            fc_input_size = self.gru_hidden_size + self.static_features_len
            self.fc1 = nn.Linear(fc_input_size, 256)
        
        # Shared layers
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        # Dueling streams
        if use_noisy:
            self.value_fc = NoisyLinear(256, 128)
            self.value = NoisyLinear(128, 1)
            self.advantage_fc = NoisyLinear(256, 128)
            self.advantage = NoisyLinear(128, action_size)
        else:
            self.value_fc = nn.Linear(256, 128)
            self.value = nn.Linear(128, 1)
            self.advantage_fc = nn.Linear(256, 128)
            self.advantage = nn.Linear(128, action_size)
    
    def forward(self, state):
        if hasattr(self, 'use_gru') and self.use_gru:
            batch_size = state.size(0)
            
            # Split into time-series and static
            ts_flat = state[:, :self.time_series_len]
            static_data = state[:, self.time_series_len:]
            
            # Reshape for GRU: (batch, sequence, features)
            ts_reshaped = ts_flat.view(batch_size, self.window_size, self.num_window_features)
            
            # GRU forward
            gru_out, _ = self.gru(ts_reshaped)
            gru_last = gru_out[:, -1, :]  # Last timestep
            
            # Combine with static features
            x = torch.cat([gru_last, static_data], dim=1)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(state))
        
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Dueling: V(s) + A(s,a) - mean(A)
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        
        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage(adv)
        
        return val + (adv - adv.mean(dim=1, keepdim=True))
    
    def reset_noise(self):
        if self.use_noisy:
            self.value_fc.reset_noise()
            self.value.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage.reset_noise()
