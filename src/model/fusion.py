"""Fusion module for combining text and tabular feature embeddings.

Supports two fusion strategies:
1. Concatenation: Simple concatenation followed by projection
2. Attention-based: Cross-attention mechanism for adaptive fusion
"""

import torch
import torch.nn as nn



class ConcatFusion(nn.Module):
    """Concatenation-based fusion (unchanged)."""
    def __init__(self, text_dim, feature_dim, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim + feature_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_size = fusion_dim

    def forward(self, text_emb, feature_emb):
        combined = torch.cat([text_emb, feature_emb], dim=1)
        return self.projection(combined)


class GatedFusion(nn.Module):
    """Gated fusion: learns when to trust features vs BERT.

    The gate is conditioned on the feature values themselves,
    so it opens/closes based on how informative the pattern
    features are for a given sample. This prevents BERT from
    always drowning out stat/pattern features.

    Architecture:
        gate = sigmoid(Linear(feature_emb))       # (B, text_dim)
        output = text_emb * gate + text_emb       # residual: always keep BERT
               + feature_proj * (1 - gate)        # add features proportionally
        → LayerNorm → Dropout → Linear → output
    """

    def __init__(self, text_dim, feature_dim, fusion_dim=256, dropout=0.3):
        super().__init__()
        # Project features to text space for gating
        self.feature_proj = nn.Linear(feature_dim, text_dim)

        # Gate: conditioned on feature values
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, text_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Dropout(dropout),
            nn.Linear(text_dim, fusion_dim),
            nn.ReLU(),
        )
        self.output_size = fusion_dim

    def forward(self, text_emb, feature_emb):
        """
        Args:
            text_emb:    (B, text_dim)   — BERT [CLS]
            feature_emb: (B, feature_dim) — stat+pattern encoded
        Returns:
            fused: (B, fusion_dim)
        """
        gate_weight = self.gate(feature_emb)          # (B, text_dim) in [0,1]
        feat_proj = self.feature_proj(feature_emb)    # (B, text_dim)

        # Residual gate: text is baseline, features modulate it
        fused = text_emb + gate_weight * feat_proj    # (B, text_dim)
        return self.output_proj(fused)


class AttentionFusion(nn.Module):
    """Attention-based fusion (kept for backward compat, use GatedFusion instead)."""

    def __init__(self, text_dim, feature_dim, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.feature_proj = nn.Linear(feature_dim, fusion_dim)
        self.attention = nn.Sequential(nn.Linear(fusion_dim, 1))
        self.output_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )
        self.output_size = fusion_dim

    def forward(self, text_emb, feature_emb):
        text_proj = torch.relu(self.text_proj(text_emb))
        feat_proj = torch.relu(self.feature_proj(feature_emb))
        stacked = torch.stack([text_proj, feat_proj], dim=1)
        attn_scores = self.attention(stacked)
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)
        return self.output_proj(fused)
