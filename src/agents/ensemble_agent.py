import torch
import numpy as np
from src.agents.multimodal_agent import MultiModalRLAgent

class EnsembleAgent:
    """
    Ensemble of 3 Agents:
    1. Aggressive (High Risk, High Reward)
    2. Conservative (Low Risk, Capital Preservation)
    3. Balanced (Standard)
    """
    def __init__(self, time_series_dim, vision_channels, action_dim, device="cuda"):
        self.device = device
        self.action_dim = action_dim
        
        # Initialize 3 Agents
        self.aggressive = MultiModalRLAgent(time_series_dim, vision_channels, action_dim, device, use_per=True)
        self.conservative = MultiModalRLAgent(time_series_dim, vision_channels, action_dim, device, use_per=True)
        self.balanced = MultiModalRLAgent(time_series_dim, vision_channels, action_dim, device, use_per=True)
        
        # Customize Hyperparameters
        # Aggressive: Higher Epsilon (Exploration), Lower Gamma (Short-term focus)
        self.aggressive.epsilon_min = 0.1
        self.aggressive.gamma = 0.90
        
        # Conservative: Lower Epsilon (Exploitation), Higher Gamma (Long-term focus)
        self.conservative.epsilon_min = 0.01
        self.conservative.gamma = 0.995
        
        # Balanced: Standard
        self.balanced.epsilon_min = 0.05
        self.balanced.gamma = 0.99
        
        self.agents = [self.aggressive, self.conservative, self.balanced]

    def act(self, ts_state, text_ids, text_mask, eval_mode=False):
        """
        Voting Mechanism:
        - If all 3 agree, take that action.
        - If 2 agree, take majority.
        - If all disagree, take Balanced action.
        """
        votes = []
        for agent in self.agents:
            votes.append(agent.act(ts_state, text_ids, text_mask, eval_mode))
            
        # Majority Vote
        counts = np.bincount(votes, minlength=self.action_dim)
        if np.max(counts) >= 2:
            return np.argmax(counts)
        else:
            # Tie-breaker: Balanced Agent
            return votes[2]

    def batch_act(self, ts_state, text_ids, text_mask):
        """
        Batch inference with voting for training loop.
        """
        batch_size = ts_state.shape[0]
        all_actions = []
        
        for agent in self.agents:
            # Epsilon Greedy for this agent
            if np.random.random() < agent.epsilon:
                actions = torch.randint(0, self.action_dim, (batch_size,)).to(self.device)
            else:
                with torch.no_grad():
                    q_vals = agent.policy_net(ts_state, text_ids, text_mask)
                    actions = q_vals.argmax(dim=1)
            all_actions.append(actions)
            
        # Stack: (3, Batch)
        stacked_actions = torch.stack(all_actions)
        
        # Voting per sample in batch
        final_actions = []
        for i in range(batch_size):
            votes = stacked_actions[:, i].cpu().numpy()
            counts = np.bincount(votes, minlength=self.action_dim)
            if np.max(counts) >= 2:
                final_actions.append(np.argmax(counts))
            else:
                # Tie-breaker: Balanced Agent (Index 2)
                final_actions.append(votes[2])
                
        return torch.LongTensor(final_actions).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in ALL agents
        # Note: In a real ensemble, you might want diversity in data too.
        # But for now, we train them on the same data to learn different policies.
        for agent in self.agents:
            agent.remember(state, action, reward, next_state, done)

    def train_step(self):
        losses = {}
        # Train all agents
        l_agg = self.aggressive.train_step()
        l_con = self.conservative.train_step()
        l_bal = self.balanced.train_step()
        
        if l_agg is not None: losses['aggressive'] = l_agg
        if l_con is not None: losses['conservative'] = l_con
        if l_bal is not None: losses['balanced'] = l_bal
        
        return losses
            
    def save(self, path_prefix):
        torch.save(self.aggressive.policy_net.state_dict(), f"{path_prefix}_aggressive.pth")
        torch.save(self.conservative.policy_net.state_dict(), f"{path_prefix}_conservative.pth")
        torch.save(self.balanced.policy_net.state_dict(), f"{path_prefix}_balanced.pth")

    def load(self, path_prefix):
        try:
            # Map to device (CPU if CUDA not available)
            map_loc = self.device
            
            self.aggressive.policy_net.load_state_dict(torch.load(f"{path_prefix}_aggressive.pth", map_location=map_loc))
            self.conservative.policy_net.load_state_dict(torch.load(f"{path_prefix}_conservative.pth", map_location=map_loc))
            self.balanced.policy_net.load_state_dict(torch.load(f"{path_prefix}_balanced.pth", map_location=map_loc))
            print("✅ Ensemble weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Ensemble weights not found. Starting fresh.")
