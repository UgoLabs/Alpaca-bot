"""
Replay Buffers for Experience Replay
"""
import numpy as np
import random
import pickle
import os
from collections import deque


class ReplayBuffer:
    """Simple FIFO replay buffer."""
    
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)


class SumTree:
    """Sum tree for prioritized sampling."""
    
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
    """Prioritized Experience Replay buffer."""
    
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
    
    def __len__(self):
        return self.tree.n_entries
