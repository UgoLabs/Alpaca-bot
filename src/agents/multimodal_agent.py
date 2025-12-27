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
        self.learning_rate = 5e-5 # Lower LR for Transformer/BERT
        self.batch_size = 32
        self.update_target_every = 1000
        self.step_count = 0
        
        # Initialize Multi-Modal Network
        self.policy_net = MultiModalAgent(time_series_dim, vision_channels, action_dim).to(device)
        self.target_net = MultiModalAgent(time_series_dim, vision_channels, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Memory
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=10000)
            self.beta = 0.4
            self.beta_increment = 0.00001
        else:
            self.memory = deque(maxlen=10000)

    def act(self, ts_state, text_ids, text_mask, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # Prepare inputs
            ts_state = torch.FloatTensor(ts_state).unsqueeze(0).to(self.device)
            text_ids = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
            text_mask = torch.LongTensor(text_mask).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(ts_state, text_ids, text_mask)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        # state is tuple: (ts_data, text_ids, text_mask)
        if self.use_per:
            cast(PrioritizedReplayBuffer, self.memory).add(state, action, reward, next_state, done)
        else:
            cast(deque, self.memory).append((state, action, reward, next_state, done))

    def train_step(self):
        # Check if enough samples
        if self.use_per:
            mem_per = cast(PrioritizedReplayBuffer, self.memory)
            if len(mem_per) < self.batch_size:
                return None
            batch, idxs, is_weights = mem_per.sample(self.batch_size, self.beta)
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
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            cast(PrioritizedReplayBuffer, self.memory).update_priorities(idxs, td_errors)
            
            # Weighted Loss
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Target Update
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
