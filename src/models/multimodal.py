import torch
import torch.nn as nn
from src.models.transformer import TransformerDQN
from src.models.vision import VisionHead
from src.models.text import TextHead

class MultiModalAgent(nn.Module):
    """
    The '3-Headed Monster' Agent.
    Combines:
    1. Time-Series Head (Transformer) -> Price/Volume sequences
    2. Vision Head (1D ResNet) -> Chart patterns
    3. Text Head (DistilBERT) -> News/Sentiment
    """
    def __init__(self, 
                 time_series_input_dim, 
                 vision_input_channels, 
                 action_dim=3, 
                 d_model=128):
        super(MultiModalAgent, self).__init__()
        
        # 1. Time-Series Head (Transformer)
        # We modify the original TransformerDQN to output embeddings instead of Q-values directly
        # Or we just use the encoder part. Let's use the encoder part.
        self.ts_head = TransformerDQN(time_series_input_dim, d_model=d_model, action_dim=action_dim)
        # We will intercept the output before the final Q-layers in the forward pass
        # Actually, let's just use the components.
        
        # 2. Vision Head
        self.vision_head = VisionHead(input_channels=vision_input_channels, output_dim=d_model)
        
        # 3. Text Head
        self.text_head = TextHead(output_dim=d_model)
        
        # Fusion Layer
        # Concatenate: TS(128) + Vision(128) + Text(128) = 384
        fusion_dim = d_model * 3
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Final Dueling Heads
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, ts_data, text_input_ids, text_attention_mask):
        """
        ts_data: (Batch, Window, Features)
        text_input_ids: (Batch, Seq_Len)
        text_attention_mask: (Batch, Seq_Len)
        """
        
        # 1. Get Time-Series Embedding
        # We need to extract the embedding from the TS head. 
        # The current TransformerDQN returns Q-values. 
        # Let's assume we modify it or just use the encoder here.
        # For cleaner code, let's just call the encoder directly here.
        
        # TS Head Forward Logic (Replicated from TransformerDQN for flexibility)
        x_ts = ts_data.permute(1, 0, 2) # (Seq, Batch, Feat)
        x_ts = self.ts_head.embedding(x_ts) * torch.sqrt(torch.tensor(self.ts_head.d_model, device=ts_data.device))
        x_ts = self.ts_head.pos_encoder(x_ts)
        ts_out = self.ts_head.transformer_encoder(x_ts)
        ts_emb = ts_out[-1, :, :] # (Batch, d_model)
        
        # 2. Get Vision Embedding
        vision_emb = self.vision_head(ts_data) # (Batch, d_model)
        
        # 3. Get Text Embedding
        text_emb = self.text_head(text_input_ids, text_attention_mask) # (Batch, d_model)
        
        # 4. Fusion
        combined = torch.cat([ts_emb, vision_emb, text_emb], dim=1)
        fused = self.fusion_layer(combined)
        
        # 5. Dueling Output
        val = self.value_stream(fused)
        adv = self.advantage_stream(fused)
        
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_vals
