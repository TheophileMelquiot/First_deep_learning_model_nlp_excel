"""Tests for the data generation and dataset modules."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generator import (
    generate_column_sample,
    generate_dataset,
    _compute_stats,
    _compute_patterns,
)
from src.data.dataset import ColumnDataset

TEST_MODEL_PATH = str(Path(__file__).resolve().parent / "fixtures" / "test-distilbert")


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def test_generate_single_sample(self):
        """Each sample should have all required keys."""
        sample = generate_column_sample("email", num_values=10)
        assert "header" in sample
        assert "values" in sample
        assert "stats" in sample
        assert "patterns" in sample
        assert "label" in sample
        assert sample["label"] == "email"
        assert len(sample["values"]) == 10

    def test_generate_all_types(self):
        """Should generate samples for all column types."""
        types = ["email", "phone", "price", "id", "date", "name", "address", "categorical"]
        for col_type in types:
            sample = generate_column_sample(col_type)
            assert sample["label"] == col_type
            assert len(sample["values"]) > 0

    def test_generate_dataset(self):
        """Full dataset should contain correct number of samples."""
        dataset = generate_dataset(num_samples_per_class=10, seed=42)
        assert len(dataset) == 10 * 8  # 10 per class * 8 classes

    def test_reproducibility(self):
        """Same seed should produce identical datasets."""
        d1 = generate_dataset(num_samples_per_class=5, seed=123)
        d2 = generate_dataset(num_samples_per_class=5, seed=123)
        for s1, s2 in zip(d1, d2):
            assert s1["header"] == s2["header"]
            assert s1["values"] == s2["values"]
            assert s1["label"] == s2["label"]

    def test_compute_stats(self):
        """Statistical features should have correct keys and valid values."""
        values = ["hello", "world", "", "test"]
        stats = _compute_stats(values)
        assert "n_unique" in stats
        assert "entropy" in stats
        assert "null_ratio" in stats
        assert "mean_length" in stats
        assert stats["n_unique"] >= 0
        assert stats["entropy"] >= 0
        assert 0.0 <= stats["null_ratio"] <= 1.0
        assert stats["mean_length"] >= 0

    def test_compute_patterns(self):
        """Pattern features should be ratios in [0, 1]."""
        values = ["test@example.com", "hello", "123"]
        patterns = _compute_patterns(values)
        for key in ["is_email", "is_phone", "is_numeric", "is_date", "has_at", "has_dot"]:
            assert key in patterns
            assert 0.0 <= patterns[key] <= 1.0

    def test_email_patterns(self):
        """Email samples should have high is_email pattern ratio."""
        sample = generate_column_sample("email", num_values=20, null_probability=0.0)
        assert sample["patterns"]["is_email"] > 0.5
        assert sample["patterns"]["has_at"] > 0.5


class TestColumnDataset:
    """Tests for the PyTorch Dataset class."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        return generate_dataset(num_samples_per_class=5, seed=42)

    def test_dataset_creation(self, small_dataset):
        """Dataset should be created without errors."""
        ds = ColumnDataset(small_dataset, tokenizer_name=TEST_MODEL_PATH, max_seq_length=32)
        assert len(ds) == len(small_dataset)

    def test_dataset_item_shapes(self, small_dataset):
        """Each item should have correct tensor shapes."""
        ds = ColumnDataset(small_dataset, tokenizer_name=TEST_MODEL_PATH, max_seq_length=32)
        item = ds[0]

        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["stat_features"].shape == (4,)
        assert item["pattern_features"].shape == (6,)
        assert item["label"].shape == ()

    def test_dataset_label_range(self, small_dataset):
        """Labels should be valid class indices."""
        ds = ColumnDataset(small_dataset, tokenizer_name=TEST_MODEL_PATH, max_seq_length=32)
        num_classes = len(ds.label_encoder.classes_)
        for i in range(len(ds)):
            label = ds[i]["label"].item()
            assert 0 <= label < num_classes

    def test_stat_normalization(self, small_dataset):
        """Statistical features should be normalized (mean ~0, std ~1)."""
        ds = ColumnDataset(small_dataset, tokenizer_name=TEST_MODEL_PATH, max_seq_length=32)
        # Check that normalization was applied
        assert ds.stat_means is not None
        assert ds.stat_stds is not None
        # Normalized features should have reasonable range
        mean_val = ds.stat_features.mean()
        assert abs(mean_val) < 1.0

    def test_shared_label_encoder(self, small_dataset):
        """Validation dataset should reuse training label encoder."""
        train_ds = ColumnDataset(small_dataset[:30], tokenizer_name=TEST_MODEL_PATH, max_seq_length=32)
        val_ds = ColumnDataset(
            small_dataset[30:],
            tokenizer_name=TEST_MODEL_PATH,
            max_seq_length=32,
            label_encoder=train_ds.label_encoder,
            stat_means=train_ds.stat_means,
            stat_stds=train_ds.stat_stds,
        )
        # Same classes
        assert list(train_ds.label_encoder.classes_) == list(val_ds.label_encoder.classes_)
