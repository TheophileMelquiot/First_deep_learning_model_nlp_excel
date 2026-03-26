"""Tests for the training pipeline."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Tests for the early stopping mechanism."""

    def test_no_stop_with_improving_loss(self):
        """Should not stop if loss keeps improving."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        model = torch.nn.Linear(10, 2)

        assert not es(1.0, model)
        assert not es(0.9, model)
        assert not es(0.8, model)
        assert not es.should_stop

    def test_stop_after_patience(self):
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        model = torch.nn.Linear(10, 2)

        es(1.0, model)  # Best
        es(1.1, model)  # No improvement (1)
        es(1.2, model)  # No improvement (2)
        stopped = es(1.3, model)  # No improvement (3) -> stop

        assert stopped
        assert es.should_stop

    def test_counter_resets_on_improvement(self):
        """Counter should reset when loss improves."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        model = torch.nn.Linear(10, 2)

        es(1.0, model)
        es(1.1, model)  # No improvement (1)
        es(1.2, model)  # No improvement (2)
        es(0.5, model)  # Improvement -> reset
        assert es.counter == 0

    def test_best_model_state_saved(self):
        """Should save the best model state."""
        es = EarlyStopping(patience=3, min_delta=0.001)
        model = torch.nn.Linear(10, 2)

        es(1.0, model)
        assert es.best_model_state is not None

    def test_min_delta_threshold(self):
        """Improvement less than min_delta should not count."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        model = torch.nn.Linear(10, 2)

        es(1.0, model)
        es(0.95, model)  # Improvement < 0.1, should not count
        assert es.counter == 1
