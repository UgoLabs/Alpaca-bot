import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
from collections import deque

# =============================================================================
# Noisy Linear Layer - For Better Exploration
# =============================================================================
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

# =============================================================================
# Dueling DQN Network (PyTorch)
# =============================================================================
class DuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, use_noisy=True):
        super(DuelingNetwork, self).__init__()
        self.use_noisy = use_noisy
        
        # Define Time Series Structure (Must match utils.py)
        self.window_size = 20
        self.num_window_features = 11
        self.time_series_len = self.window_size * self.num_window_features
        self.static_features_len = state_size - self.time_series_len
        
        if self.static_features_len < 0:
             # Fallback if state size doesn't match window assumption
             # Just use pure Dense network if dimensions are wrong
             self.use_gru = False
             print(f"Warning: State size {state_size} too small for GRU window {self.window_size}. Using Dense.")
             self.fc1 = nn.Linear(state_size, 256)
        else:
            self.use_gru = True
            # GRU Layer for Time Series
            self.gru_hidden_size = 128
            self.gru = nn.GRU(
                input_size=self.num_window_features, 
                hidden_size=self.gru_hidden_size, 
                num_layers=1, 
                batch_first=True
            )
            
            # Input to FC = GRU_Hidden + Static_Features
            fc_input_size = self.gru_hidden_size + self.static_features_len
            self.fc1 = nn.Linear(fc_input_size, 256)
            
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        if use_noisy:
            # Value stream
            self.value_fc = NoisyLinear(256, 128)
            self.value = NoisyLinear(128, 1)
            
            # Advantage stream
            self.advantage_fc = NoisyLinear(256, 128)
            self.advantage = NoisyLinear(128, action_size)
        else:
            # Value stream
            self.value_fc = nn.Linear(256, 128)
            self.value = nn.Linear(128, 1)
            
            # Advantage stream
            self.advantage_fc = nn.Linear(256, 128)
            self.advantage = nn.Linear(128, action_size)
            
    def forward(self, state):
        if hasattr(self, 'use_gru') and self.use_gru:
            # Split state into Time Series and Static
            batch_size = state.size(0)
            
            # Extract time series part: (Batch, 220)
            ts_flat = state[:, :self.time_series_len]
            
            # Extract static part: (Batch, 11)
            static_data = state[:, self.time_series_len:]
            
            # Reshape Time Series for GRU: (Batch, Sequence, Features) -> (Batch, 20, 11)
            ts_reshaped = ts_flat.view(batch_size, self.window_size, self.num_window_features)
            
            # Output: (Batch, Sequence, Hidden)
            gru_out, _ = self.gru(ts_reshaped)
            
            # Take last time step's hidden state: (Batch, 128)
            gru_last = gru_out[:, -1, :]
            
            # Combine with static features
            x = torch.cat([gru_last, static_data], dim=1)
            
            # Pass to Dense Layers
            x = F.relu(self.fc1(x))
        else:
            # Fallback legacy dense forward
            x = F.relu(self.fc1(state))
            
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        val = F.relu(self.value_fc(x))
        val = self.value(val)
        
        adv = F.relu(self.advantage_fc(x))
        adv = self.advantage(adv)
        
        # Combine: Q = V + (A - mean(A))
        return val + (adv - adv.mean(dim=1, keepdim=True))
    
    def reset_noise(self):
        if self.use_noisy:
            self.value_fc.reset_noise()
            self.value.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage.reset_noise()

# =============================================================================
# Agent Wrapper
# =============================================================================
class DuelingDQN:
    def __init__(self, state_size, action_size, learning_rate=0.0003, use_noisy=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_noisy = use_noisy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DuelingNetwork(state_size, action_size, use_noisy).to(self.device)
        self.target_model = DuelingNetwork(state_size, action_size, use_noisy).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.HuberLoss()
        
    def act(self, state, epsilon=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state)
        self.model.train()
        
        if self.use_noisy:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            if np.random.rand() <= epsilon:
                return random.randrange(self.action_size)
            return np.argmax(q_values.cpu().data.numpy())

    def train(self, states, targets, weights=None):
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Reset noise for training step
        if self.use_noisy:
            self.model.reset_noise()
            self.target_model.reset_noise()
            
        predictions = self.model(states)
        
        # We only care about the Q-values for the actions taken
        # But here 'targets' is already the full target Q-vector from the training loop
        # So we can just do MSE/Huber on the full vector or masked
        # In the training loop we'll pass the full target vector
        
        if weights is not None:
            loss = torch.mean(weights * self.loss_fn(predictions, targets))
        else:
            loss = self.loss_fn(predictions, targets)
            
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def soft_update_target_model(self, tau=0.005):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            print(f"Model file not found: {filepath}")

# =============================================================================
# Replay Buffers (Unchanged logic, just numpy storage)
# =============================================================================
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, max_size=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(max_size)
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01

    def add(self, experience, error=100.0):
        p = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, experience)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size
        
        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            indices.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            
    def size(self):
        return self.tree.n_entries
        
    def save(self, filepath):
        # Complex object, might not pickle well directly if too large
        # Just save data and tree arrays
        pass 
    
    def load(self, filepath):
        pass

class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, max_size=10000, n_step=3, gamma=0.99):
        super().__init__(max_size)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(self, experience, error=None):
        # experience: (state, action, reward, next_state, done)
        self.n_step_buffer.append(experience)
        
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.n_step_buffer[0]
            
            # Calculate n-step reward
            for i in range(1, self.n_step):
                r = self.n_step_buffer[i][2]
                reward += (self.gamma ** i) * r
                if self.n_step_buffer[i][4]: # If done
                    done = True
                    next_state = self.n_step_buffer[i][3]
                    break
            else:
                next_state = self.n_step_buffer[-1][3]
            
            super().add((state, action, reward, next_state, done))
            
        if experience[4]: # If done, clear buffer
            self.n_step_buffer.clear()
