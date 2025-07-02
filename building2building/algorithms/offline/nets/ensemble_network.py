import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from building2building.algorithms.offline.nets.ensemble_linear import EnsembleLinear


class EnsembleNetwork(nn.Module):
    """Ensemble network built using EnsembleLinear layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (200, 200, 200, 200),
        num_ensemble: int = 7,
        num_elites: int = 5,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self.device = torch.device(device)
        self.output_dim = output_dim
        
        # Build ensemble layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layer = EnsembleLinear(prev_dim, hidden_dim, num_ensemble)
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Output layer - predict both mean and log variance
        self.output_layer = EnsembleLinear(prev_dim, output_dim * 2, num_ensemble)
        
        # Initialize elite indices
        self.register_buffer('elites', torch.arange(num_elites))
        
        # Learnable variance bounds
        self.max_logvar = nn.Parameter(torch.ones(1, output_dim) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, output_dim) * -10)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble."""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Forward through hidden layers
        for layer in self.layers:
            x = torch.relu(layer(x))
        
        # Output layer
        output = self.output_layer(x)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(output, 2, dim=-1)
        
        # Bound the log variance
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
        return mean, logvar
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        """Sample random elite indices for each batch element."""
        return np.random.choice(self.elites.cpu().numpy(), size=batch_size)
    
    def set_elites(self, indices: List[int]):
        """Set elite model indices."""
        self.elites.data = torch.tensor(indices, device=self.device)
    
    def update_save(self, indices: List[int]):
        """Save model snapshots for specified indices."""
        for layer in self.layers:
            layer.update_save(indices)
        self.output_layer.update_save(indices)
    
    def load_save(self):
        """Load saved model snapshots."""
        for layer in self.layers:
            layer.load_save()
        self.output_layer.load_save()
    
    def get_decay_loss(self) -> torch.Tensor:
        """Get L2 regularization loss."""
        decay_loss = torch.tensor(0.0, device=self.device)
        for layer in self.layers:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss 