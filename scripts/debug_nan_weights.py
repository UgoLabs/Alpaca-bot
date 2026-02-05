"""Quick debug script to check model weights for NaN."""
import torch
import sys
sys.path.insert(0, '.')

from src.agents.ensemble_agent import EnsembleAgent

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the ep2 model
model_path = "models/sharpe_gen3_ep2"
agent = EnsembleAgent(time_series_dim=25, vision_channels=25, action_dim=3, device=device)
agent.load(model_path)

# Check for NaN weights
print("Checking for NaN in model weights...")
for i, (name, sub_agent) in enumerate(zip(['Aggressive', 'Conservative', 'Balanced'], agent.agents)):
    nan_count = 0
    total_params = 0
    for pname, param in sub_agent.policy_net.named_parameters():
        total_params += param.numel()
        if torch.isnan(param).any():
            nan_count += torch.isnan(param).sum().item()
            print(f"  {name}.{pname}: {torch.isnan(param).sum().item()} NaN values")
    print(f"{name}: {nan_count}/{total_params} NaN params")
