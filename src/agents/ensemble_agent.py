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

    def act(self, ts_state, text_ids, text_mask, eval_mode=False, return_q=False):
        """
        Voting Mechanism:
        - If all 3 agree, take that action.
        - If 2 agree, take majority.
        - If all disagree, take Balanced action.
        
        return_q: if True, returns (action, weighted_avg_q_value_of_chosen_action)
        """
        # Ensure inputs are Tensors on correct device
        if isinstance(ts_state, np.ndarray):
            ts_state = torch.from_numpy(ts_state).float().to(self.device)
        elif isinstance(ts_state, torch.Tensor):
            ts_state = ts_state.to(self.device)
            
        if isinstance(text_ids, torch.Tensor):
            text_ids = text_ids.to(self.device)
        if isinstance(text_mask, torch.Tensor):
            text_mask = text_mask.to(self.device)

        # Add Batch Dimension if missing
        if ts_state.dim() == 2:
            ts_state = ts_state.unsqueeze(0)
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)
        if text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)

        votes = []
        q_sums = np.zeros(self.action_dim)
        
        with torch.inference_mode():
             for agent in self.agents:
                # Forward pass
                q_vals_tensor = agent.policy_net(ts_state, text_ids, text_mask)
                q_vals = q_vals_tensor.squeeze(0) # Remove batch dim -> (Action_Dim)
                
                action = torch.argmax(q_vals).item()
                votes.append(action)
                q_sums += q_vals.cpu().numpy()

        # Majority Vote
        counts = np.bincount(votes, minlength=self.action_dim)
        if np.max(counts) >= 2:
            final_action = np.argmax(counts)
        else:
            # Tie-breaker: Balanced Agent
            final_action = votes[2]
            
        if return_q:
            # Return Q-value average of the chosen action to represent "Confidence"
            avg_q = q_sums[final_action] / 3.0
            return final_action, avg_q
            
        return final_action

    def batch_act(self, ts_state, text_ids, text_mask, in_position=None, sell_bias=0.15):
        """
        Batch inference with CONTEXT-AWARE voting.
        
        Improvements:
        1. Aggressive agent has more weight on BUY decisions
        2. Conservative agent has more weight on SELL decisions  
        3. sell_bias: probability to randomly explore SELL action (helps learn exits)
        """
        batch_size = ts_state.shape[0]
        all_actions = []
        all_q_vals = []
        
        for agent in self.agents:
            # Epsilon Greedy with SELL BIAS
            if np.random.random() < agent.epsilon:
                # Instead of uniform random, bias toward SELL to learn exits
                rand_probs = torch.rand(batch_size, device=self.device)
                actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                # sell_bias% chance of SELL, (1-sell_bias)/2 for HOLD and BUY each
                hold_cut = float(sell_bias) + 0.425
                actions[rand_probs < float(sell_bias)] = 2  # SELL
                actions[(rand_probs >= float(sell_bias)) & (rand_probs < hold_cut)] = 0  # HOLD
                actions[rand_probs >= hold_cut] = 1  # BUY
            else:
                # Fast path: keep inference on GPU and allow mixed precision.
                # This affects action selection only; training remains FP32.
                with torch.inference_mode():
                    use_autocast = isinstance(self.device, str) and self.device.startswith("cuda")
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            q_vals = agent.policy_net(ts_state, text_ids, text_mask)
                    else:
                        q_vals = agent.policy_net(ts_state, text_ids, text_mask)
                    actions = q_vals.argmax(dim=1)
                    all_q_vals.append(q_vals)
            all_actions.append(actions)
            
        # Stack: (3, Batch) - keep on GPU
        stacked_actions = torch.stack(all_actions)
        votes = stacked_actions.permute(1, 0).contiguous()  # (Batch, 3)

        # Compute weights per sample on GPU
        weights = torch.ones((batch_size, 3), device=self.device)
        has_sell = (votes == 2).any(dim=1)
        has_buy = (votes == 1).any(dim=1)
        weights[:, 1] = torch.where(has_sell, torch.tensor(1.5, device=self.device), torch.tensor(1.0, device=self.device))
        weights[:, 0] = torch.where(has_buy, torch.tensor(1.3, device=self.device), torch.tensor(1.0, device=self.device))

        # Weighted vote counts: (Batch, action_dim)
        counts = torch.zeros((batch_size, self.action_dim), device=self.device)
        for j in range(3):
            counts.scatter_add_(1, votes[:, j].unsqueeze(1), weights[:, j].unsqueeze(1))

        max_w, argmax = counts.max(dim=1)

        # Special SELL preference rule
        cons_sell = votes[:, 1] == 2
        other_not_buy = (votes[:, 0] != 1) | (votes[:, 2] != 1)
        force_sell = cons_sell & other_not_buy

        # Default tie-breaker: Balanced agent
        final_actions = votes[:, 2].clone()
        confident = max_w >= 1.5
        final_actions[confident] = argmax[confident]
        final_actions[force_sell] = 2

        return final_actions

    def remember(self, state, action, reward, next_state, done):
        # Store experience in ALL agents
        # Note: In a real ensemble, you might want diversity in data too.
        # But for now, we train them on the same data to learn different policies.
        for agent in self.agents:
            agent.remember(state, action, reward, next_state, done)

    def freeze_feature_extractors(self):
        for agent in self.agents:
            agent.freeze_feature_extractors()

    def unfreeze_all(self):
        for agent in self.agents:
            agent.unfreeze_all()

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

            # Prefer safe/quiet weights-only loading when supported.
            try:
                agg_sd = torch.load(f"{path_prefix}_aggressive.pth", map_location=map_loc, weights_only=True)
                con_sd = torch.load(f"{path_prefix}_conservative.pth", map_location=map_loc, weights_only=True)
                bal_sd = torch.load(f"{path_prefix}_balanced.pth", map_location=map_loc, weights_only=True)
            except TypeError:
                agg_sd = torch.load(f"{path_prefix}_aggressive.pth", map_location=map_loc)
                con_sd = torch.load(f"{path_prefix}_conservative.pth", map_location=map_loc)
                bal_sd = torch.load(f"{path_prefix}_balanced.pth", map_location=map_loc)

            self.aggressive.policy_net.load_state_dict(agg_sd)
            self.conservative.policy_net.load_state_dict(con_sd)
            self.balanced.policy_net.load_state_dict(bal_sd)
            print("✅ Ensemble weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Ensemble weights not found. Starting fresh.")

    def freeze_feature_extractors(self):
        for agent in self.agents:
            agent.freeze_feature_extractors()

    def unfreeze_all(self):
        for agent in self.agents:
            agent.unfreeze_all()
