import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerDQN(nn.Module):
    """
    Transformer-based Dueling DQN for Time Series Trading.
    Replaces the old LSTM/CNN architecture with a self-attention mechanism.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, action_dim=3, dropout=0.1):
        super(TransformerDQN, self).__init__()
        
        self.d_model = d_model
        
        # 1. Input Embedding (Project features to d_model size)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. Dueling Heads
        # Value Stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Window_Size, Features)
        # Transformer expects: (Batch, Sequence_Len, Features)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer
        # Output: (Batch, Sequence_Len, d_model)
        transformer_out = self.transformer_encoder(x)
        
        # We take the output of the LAST time step as the representation of the sequence
        # (Batch, d_model)
        final_state = transformer_out[:, -1, :]
        
        # Dueling DQN Logic
        val = self.value_stream(final_state)
        adv = self.advantage_stream(final_state)
        
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_vals
        
        # Q = V + (A - mean(A))
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_vals
