"""Feature encoder for statistical and pattern-based features.

Encodes the numerical tabular features (stats + patterns) into a
learned representation via a small MLP.
"""

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Encode statistical and pattern features via an MLP.

    Architecture:
        [stat_features || pattern_features] → Linear → ReLU → Dropout → Linear → ReLU

    This transforms raw tabular features into a learned representation
    that can be fused with the text embedding.
    """

    def __init__(
        self,
        num_stat_features: int = 4,
        num_pattern_features: int = 6,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        """Initialize the feature encoder.

        Args:
            num_stat_features: Number of statistical features
            num_pattern_features: Number of pattern features
            hidden_size: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()
        input_size = num_stat_features + num_pattern_features

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_size = hidden_size

    def forward(
        self,
        stat_features: torch.Tensor,
        pattern_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the feature encoder.

        Args:
            stat_features: Statistical features, shape (batch_size, num_stat_features)
            pattern_features: Pattern features, shape (batch_size, num_pattern_features)

        Returns:
            Encoded features, shape (batch_size, hidden_size)
        """
        combined = torch.cat([stat_features, pattern_features], dim=1)
        return self.encoder(combined)
