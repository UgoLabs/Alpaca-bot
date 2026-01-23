import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Union, cast
from collections import deque
from src.models.multimodal import MultiModalAgent
from src.core.priority_replay import PrioritizedReplayBuffer

class MultiModalRLAgent:
    def __init__(self, 
                 time_series_dim, 
                 vision_channels, 
                 action_dim, 
                 device="cuda",
                 use_per=True):
        
        self.device = device
        self.action_dim = action_dim
        self.use_per = use_per
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999999  # Extremely slow decay for 50k steps * 50 episodes
        self.learning_rate = 5e-6  # Slower LR for frequent updates
        self.batch_size = 64  # Larger batch for stability
        self.update_target_every = 100  # More frequent soft updates
        self.tau = 0.005  # Soft update factor (prevents sudden changes)
        self.step_count = 0
        
        # Initialize Multi-Modal Network
        self.policy_net = MultiModalAgent(time_series_dim, vision_channels, action_dim).to(device)
        self.target_net = MultiModalAgent(time_series_dim, vision_channels, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay (L2 regularization) to prevent drift
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Memory - 100k to store diverse experiences across all symbols
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=100000)
            self.beta = 0.4
            self.beta_increment = 0.00001
        else:
            self.memory = deque(maxlen=100000)

    def act(self, ts_state, text_ids, text_mask, eval_mode=False, return_q=False):
        # In eval_mode (live trading), ALWAYS use model inference - no random exploration
        if eval_mode:
            pass  # Skip to inference below
        elif random.random() < self.epsilon:
            # Training only: random exploration
            action = random.randrange(self.action_dim)
            if return_q:
                # Training exploration - return uniform Q for gradient purposes
                dummy_q = torch.ones(self.action_dim, device=self.device) / self.action_dim
                return action, dummy_q
            return action

        # REAL MODEL INFERENCE (always used for live trading)
        with torch.inference_mode():
            # Check if already batched (tensor with batch dim)
            if isinstance(ts_state, torch.Tensor) and ts_state.dim() >= 2:
                # Already batched - use as-is
                ts_batch = ts_state.to(self.device)
                ids_batch = text_ids.to(self.device)
                mask_batch = text_mask.to(self.device)
                is_batch = ts_batch.shape[0] > 1
            else:
                # Single sample - add batch dimension
                ts_batch = torch.FloatTensor(ts_state).unsqueeze(0).to(self.device)
                ids_batch = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
                mask_batch = torch.LongTensor(text_mask).unsqueeze(0).to(self.device)
                is_batch = False

            use_autocast = isinstance(self.device, str) and self.device.startswith("cuda")
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    q_values = self.policy_net(ts_batch, ids_batch, mask_batch)
            else:
                q_values = self.policy_net(ts_batch, ids_batch, mask_batch)
            
            if is_batch:
                # Return batch of actions
                actions = q_values.argmax(dim=1)
                if return_q:
                    return actions, q_values
                return actions
            else:
                # Single sample
                action = q_values.argmax().item()
                if return_q:
                    return action, q_values.squeeze(0)
                return action

    def remember(self, state, action, reward, next_state, done):
        # state is tuple: (ts_data, text_ids, text_mask)
        if self.use_per:
            cast(PrioritizedReplayBuffer, self.memory).add(state, action, reward, next_state, done)
        else:
            cast(deque, self.memory).append((state, action, reward, next_state, done))

    def freeze_feature_extractors(self):
        """Freeze Time-Series and Vision backbones, only train Fusion/Heads."""
        print("❄️ Freezing Feature Extractors (TS + Vision + Text)...")
        # Freeze TS Head (Transformer/Encoder)
        for param in self.policy_net.ts_head.parameters():
            param.requires_grad = False
        
        # Freeze Vision Head
        for param in self.policy_net.vision_head.parameters():
            param.requires_grad = False
            
        # Text Head
        for param in self.policy_net.text_head.parameters():
            param.requires_grad = False
            
        # Re-init optimizer to track only active parameters
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.policy_net.parameters()), 
                                     lr=self.learning_rate, weight_decay=1e-4)

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        print("🔥 Unfreezing All Layers...")
        for param in self.policy_net.parameters():
            param.requires_grad = True
            
        # Re-init optimizer for all parameters (usually with lower LR)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate * 0.1, weight_decay=1e-4)

    def train_step(self):
        # Check if enough samples
        if self.use_per:
            mem_per = cast(PrioritizedReplayBuffer, self.memory)
            if len(mem_per) < self.batch_size:
                return None
            batch, idxs, is_weights = mem_per.sample(self.batch_size, self.beta)
            # Handle empty batch (corrupted buffer)
            if not batch or len(batch) < self.batch_size:
                return None
            # Update Beta
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            mem_deque = cast(deque, self.memory)
            if len(mem_deque) < self.batch_size:
                return None
            batch = random.sample(mem_deque, self.batch_size)
            idxs, is_weights = None, None

        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Unpack States
        ts_batch = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
        text_ids_batch = torch.LongTensor(np.array([s[1] for s in states])).to(self.device)
        text_mask_batch = torch.LongTensor(np.array([s[2] for s in states])).to(self.device)
        
        # Unpack Next States
        next_ts_batch = torch.FloatTensor(np.array([s[0] for s in next_states])).to(self.device)
        next_text_ids_batch = torch.LongTensor(np.array([s[1] for s in next_states])).to(self.device)
        next_text_mask_batch = torch.LongTensor(np.array([s[2] for s in next_states])).to(self.device)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        if self.use_per:
            weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        
        # Current Q
        current_q = self.policy_net(ts_batch, text_ids_batch, text_mask_batch).gather(1, actions)
        
        # Target Q (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_ts_batch, next_text_ids_batch, next_text_mask_batch).argmax(1, keepdim=True)
            next_q = self.target_net(next_ts_batch, next_text_ids_batch, next_text_mask_batch).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Loss Calculation
        if self.use_per:
            # Element-wise loss for priority update
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().flatten()
            cast(PrioritizedReplayBuffer, self.memory).update_priorities(idxs, td_errors)
            
            # Weighted Loss
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # NOTE: Epsilon decay moved to episode level in training script
        # (was decaying too fast per-step)
            
        # Soft Target Update (prevents catastrophic forgetting)
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
