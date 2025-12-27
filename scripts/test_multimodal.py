import torch
from src.models.multimodal import MultiModalAgent

def test_model():
    print("Initializing Multi-Modal Agent...")
    
    # Dimensions
    batch_size = 4
    window_size = 60
    num_features = 11 # OHLCV + Indicators
    seq_len = 32 # Text sequence length
    
    # Instantiate Model
    model = MultiModalAgent(
        time_series_input_dim=num_features,
        vision_input_channels=num_features,
        action_dim=3,
        d_model=64 # Keep small for test
    )
    
    print("Model initialized.")
    
    # Create Dummy Data
    ts_data = torch.randn(batch_size, window_size, num_features)
    text_ids = torch.randint(0, 1000, (batch_size, seq_len))
    text_mask = torch.ones(batch_size, seq_len)
    
    print("Forward pass...")
    q_vals = model(ts_data, text_ids, text_mask)
    
    print(f"Output Shape: {q_vals.shape}")
    assert q_vals.shape == (batch_size, 3)
    print("✅ Test Passed: Multi-Modal Model is functional.")

if __name__ == "__main__":
    test_model()
