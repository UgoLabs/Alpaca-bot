import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Since Transformers have no recurrence, they need this to know 'Order'.
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

class TransformerDQN(nn.Module):
    """
    The 'Titan' Architecture.
    Uses Multi-Head Self-Attention to find dependencies across the entire time window instantly.
    """
    def __init__(self, state_size, action_size, 
                 d_model=128,      # Internal dimension (Embedding size)
                 nhead=4,          # Number of attention heads
                 num_layers=2,     # Number of transformer blocks
                 dropout=0.1,
                 window_size=60):  # How far back we look
        
        super(TransformerDQN, self).__init__()
        
        # 1. Feature Extraction (Input -> Embedding)
        # We assume input is flattened (window_size * features). We need to reshape it.
        self.window_size = window_size
        self.num_features = state_size // window_size # Approximate if evenly divisible
        
        # If static features exist, this simple reshape logic needs tweaking. 
        # For this design, we assume purely time-series input for simplicity.
        self.input_embedding = nn.Linear(self.num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)
        
        # 3. Transformer Encoder (The Brain)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Aggregation (Sequence -> Vector)
        # We can take the mean, or just the last timestep.
        # "Global Average Pooling" usually works best for classification.
        
        # 5. Dueling Heads (The Decision Maker)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
    def forward(self, state):
        # State shape: (Batch, Total_Features) -> needs (Batch, Window, Features)
        batch_size = state.size(0)
        
        # Reshape
        x = state.view(batch_size, self.window_size, self.num_features)
        
        # Transformer expects (Seq_Len, Batch, Dim)
        x = x.permute(1, 0, 2) 
        
        # Embed & Position
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        
        # Apply Transformer
        # Output: (Seq_Len, Batch, d_model)
        x = self.transformer_encoder(x)
        
        # Pool (Use the last timestep's output as the "Summary" of the sequence)
        # Or mean pool: x = x.mean(dim=0)
        final_embedding = x[-1, :, :] # (Batch, d_model)
        
        # Dueling Logic
        val = self.value_head(final_embedding)
        adv = self.advantage_head(final_embedding)
        
        return val + (adv - adv.mean(dim=1, keepdim=True))
