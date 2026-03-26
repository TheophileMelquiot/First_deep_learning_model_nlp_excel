"""Fusion module for combining text and tabular feature embeddings.

Supports two fusion strategies:
1. Concatenation: Simple concatenation followed by projection
2. Attention-based: Cross-attention mechanism for adaptive fusion
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Concatenation-based fusion of text and feature embeddings.

    Architecture:
        [text_embedding || feature_embedding] → Linear → ReLU → Dropout
    """

    def __init__(
        self,
        text_dim: int,
        feature_dim: int,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim + feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_size = fusion_dim

    def forward(
        self,
        text_emb: torch.Tensor,
        feature_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse text and feature embeddings via concatenation.

        Args:
            text_emb: Text embeddings, shape (batch_size, text_dim)
            feature_emb: Feature embeddings, shape (batch_size, feature_dim)

        Returns:
            Fused representation, shape (batch_size, fusion_dim)
        """
        combined = torch.cat([text_emb, feature_emb], dim=1)
        return self.projection(combined)


class AttentionFusion(nn.Module):
    """Attention-based fusion of text and feature embeddings.

    Uses a learned attention mechanism to weight the contribution
    of each modality (text vs features) adaptively.

    Architecture:
        Compute attention weights over [text_emb, feature_emb]
        → weighted sum → projection
    """

    def __init__(
        self,
        text_dim: int,
        feature_dim: int,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Project both modalities to same dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.feature_proj = nn.Linear(feature_dim, fusion_dim)

        # Attention scoring
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, 1),
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )
        self.output_size = fusion_dim

    def forward(
        self,
        text_emb: torch.Tensor,
        feature_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse text and feature embeddings via attention.

        Args:
            text_emb: Text embeddings, shape (batch_size, text_dim)
            feature_emb: Feature embeddings, shape (batch_size, feature_dim)

        Returns:
            Fused representation, shape (batch_size, fusion_dim)
        """
        # Project to common space
        text_proj = torch.relu(self.text_proj(text_emb))      # (B, fusion_dim)
        feat_proj = torch.relu(self.feature_proj(feature_emb)) # (B, fusion_dim)

        # Stack modalities: (B, 2, fusion_dim)
        stacked = torch.stack([text_proj, feat_proj], dim=1)

        # Compute attention weights: (B, 2, 1)
        attn_scores = self.attention(stacked)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum: (B, fusion_dim)
        fused = (stacked * attn_weights).sum(dim=1)

        return self.output_proj(fused)
