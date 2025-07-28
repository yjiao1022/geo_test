"""
Stripped-down STGCN variants for bias testing.

These minimal models help isolate whether bias comes from network complexity
or systematic pipeline issues.
"""

import torch
import torch.nn as nn
from .stgcn_model import STGCNReportingModel
from .common_utils import ReportingConfig

class STGCNShallowModel(STGCNReportingModel):
    """
    Stripped-down STGCN variant that essentially returns last observed control value.
    
    Configuration:
    - hidden_dim = 1
    - num_st_blocks = 0  
    - dropout = 0.9
    - input/output projections → identity
    
    This effectively returns the last observed control value (plus noise) as prediction.
    """
    
    def __init__(self, **kwargs):
        """Initialize with minimal configuration."""
        # Override key parameters to make model trivial
        kwargs.update({
            'hidden_dim': 1,
            'num_st_blocks': 0,
            'dropout': 0.9,
            'epochs': kwargs.get('epochs', 20),
            'verbose': kwargs.get('verbose', False),
            'reporting_config': kwargs.get('reporting_config', ReportingConfig())
        })
        super().__init__(**kwargs)
        
    def _build_model(self, n_features: int, device: str) -> nn.Module:
        """Build trivial model that's essentially identity mapping."""
        
        class TrivialSTGCN(nn.Module):
            def __init__(self, n_features, dropout=0.9):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                # Minimal linear layer that approximates identity
                self.projection = nn.Linear(n_features, n_features)
                # Initialize as near-identity
                with torch.no_grad():
                    self.projection.weight.data = torch.eye(n_features) * 0.1
                    self.projection.bias.data.zero_()
                
            def forward(self, x, edge_index=None, edge_weight=None):
                # x shape: [batch_size, seq_len, n_nodes, n_features]
                batch_size, seq_len, n_nodes, n_features = x.shape
                
                # Apply heavy dropout to make it nearly random
                x = self.dropout(x)
                
                # Reshape for linear layer
                x_reshaped = x.view(-1, n_features)
                
                # Apply minimal transformation  
                out = self.projection(x_reshaped)
                
                # Reshape back
                out = out.view(batch_size, seq_len, n_nodes, n_features)
                
                return out
                
        return TrivialSTGCN(n_features, self.dropout).to(device)


class STGCNIntermediateModel(STGCNReportingModel):
    """
    Intermediate STGCN variant with limited complexity.
    
    Configuration:
    - hidden_dim = 8
    - num_st_blocks = 1
    - dropout = 0.5
    
    Tests if bias grows with model flexibility.
    """
    
    def __init__(self, **kwargs):
        """Initialize with intermediate configuration."""
        # Override key parameters for intermediate complexity
        kwargs.update({
            'hidden_dim': 8,
            'num_st_blocks': 1,
            'dropout': 0.5,
            'epochs': kwargs.get('epochs', 20),
            'verbose': kwargs.get('verbose', False),
            'reporting_config': kwargs.get('reporting_config', ReportingConfig())
        })
        super().__init__(**kwargs)