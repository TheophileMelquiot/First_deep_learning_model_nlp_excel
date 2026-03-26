"""Shared fixtures for tests.

Provides a local test model path to avoid HuggingFace downloads
in CI/sandbox environments.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_MODEL_PATH = str(Path(__file__).resolve().parent / "fixtures" / "test-distilbert")
