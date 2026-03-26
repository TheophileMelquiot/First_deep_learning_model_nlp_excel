"""Tests for the model components."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.text_encoder import TextEncoder
from src.model.feature_encoder import FeatureEncoder
from src.model.fusion import ConcatFusion, AttentionFusion
from src.model.classifier import ColumnClassifier

TEST_MODEL_PATH = str(Path(__file__).resolve().parent / "fixtures" / "test-distilbert")


class TestTextEncoder:
    """Tests for the transformer-based text encoder."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return TextEncoder(model_name=TEST_MODEL_PATH, freeze=False)

    def test_output_shape(self, encoder):
        """Output should be (batch_size, hidden_size)."""
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        output = encoder(input_ids, attention_mask)
        assert output.shape == (2, encoder.hidden_size)

    def test_hidden_size_property(self, encoder):
        """hidden_size should match transformer config."""
        assert encoder.hidden_size == 32  # tiny test model

    def test_frozen_parameters(self):
        """Frozen encoder should have no grad on transformer params."""
        frozen = TextEncoder(model_name=TEST_MODEL_PATH, freeze=True)
        for param in frozen.transformer.parameters():
            assert not param.requires_grad


class TestFeatureEncoder:
    """Tests for the tabular feature encoder."""

    def test_output_shape(self):
        encoder = FeatureEncoder(num_stat_features=4, num_pattern_features=6, hidden_size=64)
        stats = torch.randn(2, 4)
        patterns = torch.randn(2, 6)
        output = encoder(stats, patterns)
        assert output.shape == (2, 64)

    def test_different_sizes(self):
        encoder = FeatureEncoder(num_stat_features=3, num_pattern_features=5, hidden_size=32)
        stats = torch.randn(4, 3)
        patterns = torch.randn(4, 5)
        output = encoder(stats, patterns)
        assert output.shape == (4, 32)


class TestFusion:
    """Tests for fusion modules."""

    def test_concat_fusion_shape(self):
        fusion = ConcatFusion(text_dim=768, feature_dim=64, fusion_dim=256)
        text_emb = torch.randn(2, 768)
        feat_emb = torch.randn(2, 64)
        output = fusion(text_emb, feat_emb)
        assert output.shape == (2, 256)

    def test_attention_fusion_shape(self):
        fusion = AttentionFusion(text_dim=768, feature_dim=64, fusion_dim=256)
        text_emb = torch.randn(2, 768)
        feat_emb = torch.randn(2, 64)
        output = fusion(text_emb, feat_emb)
        assert output.shape == (2, 256)


class TestColumnClassifier:
    """Tests for the full column classifier."""

    @pytest.fixture(scope="class")
    def model(self):
        return ColumnClassifier(
            num_classes=8,
            transformer_name=TEST_MODEL_PATH,
            num_stat_features=4,
            num_pattern_features=6,
            feature_hidden_size=64,
            fusion_hidden_size=256,
            dropout=0.1,
        )

    def test_forward_shape(self, model):
        """Output logits should be (batch_size, num_classes)."""
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        stats = torch.randn(2, 4)
        patterns = torch.randn(2, 6)
        logits = model(input_ids, attention_mask, stats, patterns)
        assert logits.shape == (2, 8)

    def test_softmax_output(self, model):
        """Softmax of logits should sum to 1."""
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        stats = torch.randn(2, 4)
        patterns = torch.randn(2, 6)
        logits = model(input_ids, attention_mask, stats, patterns)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_backward_pass(self, model):
        """Model should support backward pass."""
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        stats = torch.randn(2, 4)
        patterns = torch.randn(2, 6)
        labels = torch.randint(0, 8, (2,))

        logits = model(input_ids, attention_mask, stats, patterns)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        # Check that gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad
