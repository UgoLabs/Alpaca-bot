"""
Transformer-Based DQN Architecture ('The Titan')
Uses Multi-Head Self-Attention to capture complex dependencies in price history.
Optimized for RTX 4070 (Parallel Processing).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.settings import TrainingConfig

class PositionalEncoding(nn.Module):
    """
    Injects timing information. 
    Without this, the model wouldn't know that 'Bar 60' is more recent than 'Bar 1'.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Seq_Len, Batch_Size, Embedding_Dim]
        return x + self.pe[:x.size(0), :]

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for consistent exploration without Epsilon-Greedy."""
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
    Main Agent Class.
    Input: Flattened State Vector
    Internal: Transformer Encoder
    Output: Q-Values (Dueling)
    """
    def __init__(self, state_size, action_size, 
                 d_model=256,      # Larger embedding for RTX 4070
                 nhead=4,          # 4 Attention Heads
                 num_layers=2,     # 2 Transformer Blocks
                 dropout=0.1,
                 use_noisy=True):
        
        super(DuelingNetwork, self).__init__()
        
        self.window_size = TrainingConfig.WINDOW_SIZE
        self.num_window_features = TrainingConfig.NUM_WINDOW_FEATURES
        self.time_series_len = self.window_size * self.num_window_features
        self.static_features_len = state_size - self.time_series_len
        self.use_noisy = use_noisy
        
        # 1. Feature Embedding
        # Projects the raw input features (e.g., 15 dims) into d_model (256 dims)
        self.feature_embedding = nn.Linear(self.num_window_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.window_size)
        
        # 3. Transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Static Feature Processing
        if self.static_features_len > 0:
            self.static_fc = nn.Linear(self.static_features_len, 64)
            combined_dim = d_model + 64
        else:
            combined_dim = d_model
            
        # 5. Dueling Heads
        if use_noisy:
            self.value_fc = NoisyLinear(combined_dim, 256)
            self.value = NoisyLinear(256, 1)
            self.advantage_fc = NoisyLinear(combined_dim, 256)
            self.advantage = NoisyLinear(256, action_size)
        else:
            self.value_fc = nn.Linear(combined_dim, 256)
            self.value = nn.Linear(256, 1)
            self.advantage_fc = nn.Linear(combined_dim, 256)
            self.advantage = nn.Linear(256, action_size)
            
    def forward(self, state):
        batch_size = state.size(0)
        
        # Split State
        ts_flat = state[:, :self.time_series_len]
        static_data = state[:, self.time_series_len:]
        
        # Reshape TS: (Batch, Window, Features)
        ts_reshaped = ts_flat.view(batch_size, self.window_size, self.num_window_features)
        
        # Transformer Input Format: (Seq, Batch, Feature)
        x = ts_reshaped.permute(1, 0, 2)
        
        # Embed
        x = self.feature_embedding(x) # (Seq, Batch, d_model)
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer_encoder(x)
        
        # Pooling: Use the LAST time step as the "Current State Representation"
        # Since we use causal masking order (bar 60 comes last), x[-1] contains info from 0..60 refined.
        final_embedding = x[-1, :, :] # (Batch, d_model)
        
        # Combine Static
        if self.static_features_len > 0:
            static_out = F.relu(self.static_fc(static_data))
            combined = torch.cat([final_embedding, static_out], dim=1)
        else:
            combined = final_embedding
            
        # Dueling
        if self.use_noisy:
            # NoisyNet doesn't use dropout usually
            val = F.relu(self.value_fc(combined))
            val = self.value(val)
            adv = F.relu(self.advantage_fc(combined))
            adv = self.advantage(adv)
        else:
            val = F.relu(self.value_fc(combined))
            val = self.value(val)
            adv = F.relu(self.advantage_fc(combined))
            adv = self.advantage(adv)
            
        return val + (adv - adv.mean(dim=1, keepdim=True))
        
    def reset_noise(self):
        if self.use_noisy:
            self.value_fc.reset_noise()
            self.value.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage.reset_noise()
