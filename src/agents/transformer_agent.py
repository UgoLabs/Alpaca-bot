import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from src.models.transformer import TransformerDQN

class TransformerAgent:
    def __init__(self, state_dim, action_dim, device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.update_target_every = 1000
        self.step_count = 0
        
        # Models
        self.policy_net = TransformerDQN(state_dim, action_dim=action_dim).to(device)
        self.target_net = TransformerDQN(state_dim, action_dim=action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=50000)

    def act(self, state, eval_mode=False):
        """
        Select action using Epsilon-Greedy policy.
        state: (Window_Size, Features) numpy array or tensor
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
            else:
                if state.dim() == 2:
                    state = state.unsqueeze(0)
            
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q (Double DQN)
        with torch.no_grad():
            # Select best action using Policy Net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Evaluate that action using Target Net
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update Target Net
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
