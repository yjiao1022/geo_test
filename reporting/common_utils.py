"""
Common utilities for reporting models.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class ReportingConfig:
    """Configuration for reporting models."""
    use_observed_spend: bool = False
    
def apply_observed_spend_override(predictions_df, treatment_data, config):
    """Apply observed spend override if configured."""
    if config.use_observed_spend:
        # Replace predicted spend with observed spend
        predictions_df = predictions_df.copy()
        predictions_df['spend'] = treatment_data['spend'].values
    return predictions_df