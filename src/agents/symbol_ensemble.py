"""
Symbol-specific Ensemble Agent.

Loads individual models trained per symbol and uses them for voting-based decisions.
Falls back to a general model for symbols without specific training.
"""

import os
import glob
import torch
import numpy as np
from typing import Dict, Optional

from src.agents.multimodal_agent import MultimodalTradingAgent


class SymbolEnsembleAgent:
    """
    Ensemble that uses symbol-specific models when available,
    falls back to general model otherwise.
    """
    
    def __init__(
        self,
        time_series_dim: int = 11,
        vision_channels: int = 11,
        action_dim: int = 3,
        device: str = "cuda",
        symbol_models_dir: str = "models/symbol_models",
        fallback_model_path: str = "models/swing_best_phase2",
    ):
        self.device = device
        self.time_series_dim = time_series_dim
        self.vision_channels = vision_channels
        self.action_dim = action_dim
        self.symbol_models_dir = symbol_models_dir
        self.fallback_model_path = fallback_model_path
        
        # Cache loaded models
        self.symbol_agents: Dict[str, MultimodalTradingAgent] = {}
        self.fallback_agent: Optional[MultimodalTradingAgent] = None
        
        # Discover available symbol models
        self.available_symbols = self._discover_symbol_models()
        print(f"Found {len(self.available_symbols)} symbol-specific models")
        
        # Load fallback model
        self._load_fallback_model()
    
    def _discover_symbol_models(self) -> set:
        """Find all available symbol models."""
        pattern = os.path.join(self.symbol_models_dir, "*_best.pth")
        paths = glob.glob(pattern)
        symbols = set()
        for path in paths:
            basename = os.path.basename(path)
            # Extract symbol from path like "AAPL_best.pth"
            symbol = basename.replace("_best.pth", "")
            symbols.add(symbol.upper())
        return symbols
    
    def _load_fallback_model(self):
        """Load the general fallback model."""
        try:
            self.fallback_agent = MultimodalTradingAgent(
                time_series_dim=self.time_series_dim,
                vision_channels=self.vision_channels,
                action_dim=self.action_dim,
                device=self.device
            )
            self.fallback_agent.load(self.fallback_model_path)
            self.fallback_agent.epsilon = 0.0  # No exploration in production
            self.fallback_agent.policy_net.eval()
            print(f"Loaded fallback model from {self.fallback_model_path}")
        except Exception as e:
            print(f"Warning: Could not load fallback model: {e}")
            self.fallback_agent = None
    
    def _get_or_load_agent(self, symbol: str) -> Optional[MultimodalTradingAgent]:
        """Get symbol-specific agent, loading if necessary."""
        symbol = symbol.upper()
        
        # Check cache first
        if symbol in self.symbol_agents:
            return self.symbol_agents[symbol]
        
        # Check if model exists
        if symbol not in self.available_symbols:
            return None
        
        # Load model
        try:
            model_path = os.path.join(self.symbol_models_dir, f"{symbol}_best")
            agent = MultimodalTradingAgent(
                time_series_dim=self.time_series_dim,
                vision_channels=self.vision_channels,
                action_dim=self.action_dim,
                device=self.device
            )
            agent.load(model_path)
            agent.epsilon = 0.0
            agent.policy_net.eval()
            
            self.symbol_agents[symbol] = agent
            return agent
        except Exception as e:
            print(f"Warning: Could not load model for {symbol}: {e}")
            return None
    
    def get_action(
        self,
        symbol: str,
        state: np.ndarray,
        text_ids: np.ndarray,
        text_mask: np.ndarray,
    ) -> int:
        """
        Get action for a specific symbol.
        Uses symbol-specific model if available, otherwise fallback.
        """
        symbol = symbol.upper()
        
        # Try symbol-specific model first
        agent = self._get_or_load_agent(symbol)
        
        if agent is None:
            # Use fallback
            agent = self.fallback_agent
            if agent is None:
                # No model available at all, default to HOLD
                return 0
        
        # Get action from agent
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            text_ids_t = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
            text_mask_t = torch.LongTensor(text_mask).unsqueeze(0).to(self.device)
            
            action = agent.batch_act(state_t, text_ids_t, text_mask_t)
            return action[0].item()
    
    def get_actions_batch(
        self,
        symbols: list,
        states: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get actions for a batch of symbols.
        Each symbol uses its specific model if available.
        """
        actions = []
        
        for i, symbol in enumerate(symbols):
            state = states[i].cpu().numpy()
            t_ids = text_ids[i].cpu().numpy()
            t_mask = text_mask[i].cpu().numpy()
            
            action = self.get_action(symbol, state, t_ids, t_mask)
            actions.append(action)
        
        return torch.tensor(actions, device=self.device)
    
    def get_ensemble_action(
        self,
        symbol: str,
        state: np.ndarray,
        text_ids: np.ndarray,
        text_mask: np.ndarray,
        vote_threshold: float = 0.5,
    ) -> int:
        """
        Get action using voting from multiple models.
        
        Uses:
        1. Symbol-specific model (if exists)
        2. Fallback model
        3. Majority vote
        """
        votes = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        
        symbol = symbol.upper()
        
        # Prepare tensors
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        text_ids_t = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
        text_mask_t = torch.LongTensor(text_mask).unsqueeze(0).to(self.device)
        
        # Vote 1: Symbol-specific model
        symbol_agent = self._get_or_load_agent(symbol)
        if symbol_agent is not None:
            with torch.no_grad():
                action = symbol_agent.batch_act(state_t, text_ids_t, text_mask_t)[0].item()
                votes[action] += 2  # Weight symbol-specific higher
        
        # Vote 2: Fallback model
        if self.fallback_agent is not None:
            with torch.no_grad():
                action = self.fallback_agent.batch_act(state_t, text_ids_t, text_mask_t)[0].item()
                votes[action] += 1
        
        # Majority vote
        total_votes = sum(votes.values())
        if total_votes == 0:
            return 0  # Default HOLD
        
        # Get action with most votes
        best_action = max(votes, key=votes.get)
        vote_pct = votes[best_action] / total_votes
        
        # If not confident enough, HOLD
        if vote_pct < vote_threshold:
            return 0
        
        return best_action
    
    def has_symbol_model(self, symbol: str) -> bool:
        """Check if we have a specific model for this symbol."""
        return symbol.upper() in self.available_symbols
    
    def get_loaded_count(self) -> int:
        """Get number of currently loaded models."""
        return len(self.symbol_agents) + (1 if self.fallback_agent else 0)


# For backwards compatibility with EnsembleAgent interface
class HybridEnsembleAgent(SymbolEnsembleAgent):
    """Alias for SymbolEnsembleAgent with EnsembleAgent-like interface."""
    
    def batch_act(self, states, text_ids, text_mask, symbols=None):
        """
        Batch action interface compatible with existing code.
        If symbols provided, uses symbol-specific models.
        Otherwise uses fallback for all.
        """
        batch_size = states.shape[0]
        
        if symbols is None:
            # Use fallback for all
            if self.fallback_agent is not None:
                return self.fallback_agent.batch_act(states, text_ids, text_mask)
            else:
                return torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        return self.get_actions_batch(symbols, states, text_ids, text_mask)
    
    def save(self, prefix: str):
        """Save is not supported for hybrid ensemble."""
        print("Warning: HybridEnsembleAgent.save() not supported")
    
    def load(self, prefix: str):
        """Load fallback model."""
        self.fallback_model_path = prefix
        self._load_fallback_model()
