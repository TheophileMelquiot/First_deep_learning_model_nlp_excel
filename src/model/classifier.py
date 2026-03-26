"""Full column classifier model combining all components.

Architecture:
    Text (header + values) → TextEncoder (DistilBERT) → text_embedding
    Stats + Patterns → FeatureEncoder (MLP) → feature_embedding
    [text_embedding, feature_embedding] → Fusion → fused_representation
    fused_representation → ClassifierHead (MLP) → logits

Supports ablation by selectively disabling input modalities.
"""

import torch
import torch.nn as nn

from src.model.text_encoder import TextEncoder
from src.model.feature_encoder import FeatureEncoder
from src.model.fusion import ConcatFusion, AttentionFusion


class ColumnClassifier(nn.Module):
    """Multi-modal deep learning model for column type classification.

    Combines transformer-based text encoding with tabular feature
    encoding through a fusion module, followed by an MLP classifier.
    """

    def __init__(
        self,
        num_classes: int = 8,
        transformer_name: str = "distilbert-base-uncased",
        num_stat_features: int = 4,
        num_pattern_features: int = 6,
        feature_hidden_size: int = 64,
        fusion_hidden_size: int = 256,
        dropout: float = 0.3,
        freeze_transformer: bool = False,
        fusion_type: str = "concat",
        use_header: bool = True,
        use_values: bool = True,
        use_stats: bool = True,
        use_patterns: bool = True,
    ):
        """Initialize the column classifier.

        Args:
            num_classes: Number of output classes
            transformer_name: HuggingFace model name
            num_stat_features: Number of statistical features
            num_pattern_features: Number of pattern features
            feature_hidden_size: Feature encoder hidden size
            fusion_hidden_size: Fusion layer hidden size
            dropout: Dropout probability
            freeze_transformer: Whether to freeze transformer weights
            fusion_type: "concat" or "attention"
            use_header: Whether to use header text (ablation)
            use_values: Whether to use value text (ablation)
            use_stats: Whether to use statistical features (ablation)
            use_patterns: Whether to use pattern features (ablation)
        """
        super().__init__()

        self.use_header = use_header
        self.use_values = use_values
        self.use_stats = use_stats
        self.use_patterns = use_patterns

        # Determine which modalities are active
        self.use_text = use_header or use_values
        self.use_features = use_stats or use_patterns

        # Text encoder
        if self.use_text:
            self.text_encoder = TextEncoder(
                model_name=transformer_name,
                freeze=freeze_transformer,
            )
            text_dim = self.text_encoder.hidden_size
        else:
            text_dim = 0

        # Feature encoder
        actual_stat_features = num_stat_features if use_stats else 0
        actual_pattern_features = num_pattern_features if use_patterns else 0

        if self.use_features:
            self.feature_encoder = FeatureEncoder(
                num_stat_features=actual_stat_features,
                num_pattern_features=actual_pattern_features,
                hidden_size=feature_hidden_size,
                dropout=dropout,
            )
            feature_dim = self.feature_encoder.output_size
        else:
            feature_dim = 0

        # Fusion
        if self.use_text and self.use_features:
            if fusion_type == "attention":
                self.fusion = AttentionFusion(
                    text_dim=text_dim,
                    feature_dim=feature_dim,
                    fusion_dim=fusion_hidden_size,
                    dropout=dropout,
                )
            else:
                self.fusion = ConcatFusion(
                    text_dim=text_dim,
                    feature_dim=feature_dim,
                    fusion_dim=fusion_hidden_size,
                    dropout=dropout,
                )
            classifier_input_dim = self.fusion.output_size
        elif self.use_text:
            classifier_input_dim = text_dim
        elif self.use_features:
            classifier_input_dim = feature_dim
        else:
            raise ValueError("At least one modality must be enabled")

        # Classifier head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        stat_features: torch.Tensor | None = None,
        pattern_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)
            stat_features: Statistical features, shape (batch_size, num_stat_features)
            pattern_features: Pattern features, shape (batch_size, num_pattern_features)

        Returns:
            Logits, shape (batch_size, num_classes)
        """
        # Text encoding
        text_emb = None
        if self.use_text and input_ids is not None:
            text_emb = self.text_encoder(input_ids, attention_mask)

        # Feature encoding
        feature_emb = None
        if self.use_features:
            feat_parts = []
            if self.use_stats and stat_features is not None:
                feat_parts.append(stat_features)
            if self.use_patterns and pattern_features is not None:
                feat_parts.append(pattern_features)

            if feat_parts:
                if len(feat_parts) == 1:
                    combined_features = feat_parts[0]
                else:
                    combined_features = torch.cat(feat_parts, dim=1)
                # Pass dummy zero tensors for missing modalities
                if self.use_stats and stat_features is not None and self.use_patterns and pattern_features is not None:
                    feature_emb = self.feature_encoder(stat_features, pattern_features)
                elif self.use_stats and stat_features is not None:
                    feature_emb = self.feature_encoder(stat_features, torch.zeros(stat_features.size(0), 0, device=stat_features.device))
                elif self.use_patterns and pattern_features is not None:
                    feature_emb = self.feature_encoder(torch.zeros(pattern_features.size(0), 0, device=pattern_features.device), pattern_features)

        # Fusion
        if text_emb is not None and feature_emb is not None:
            fused = self.fusion(text_emb, feature_emb)
        elif text_emb is not None:
            fused = text_emb
        elif feature_emb is not None:
            fused = feature_emb
        else:
            raise ValueError("No valid input provided to the model")

        # Classification
        logits = self.classifier(fused)
        return logits
