"""Custom PyTorch Dataset and DataLoader for column classification.

Handles tokenization of text features (header + values) using a
pretrained transformer tokenizer, normalization of numerical features,
and label encoding.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Any


class ColumnDataset(Dataset):
    """PyTorch Dataset for tabular column classification.

    Each sample consists of:
    - Tokenized text (header + concatenated values)
    - Statistical feature vector
    - Pattern feature vector
    - Integer label
    """

    STAT_KEYS = ["n_unique", "entropy", "null_ratio", "mean_length"]
    PATTERN_KEYS = ["is_email", "is_phone", "is_numeric", "is_date", "has_at", "has_dot", "is_text", "has_space", "has_digit"]

    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer_name: str = "distilbert-base-uncased",
        max_seq_length: int = 128,
        label_encoder: LabelEncoder | None = None,
        stat_means: np.ndarray | None = None,
        stat_stds: np.ndarray | None = None,
    ):
        """Initialize the dataset.

        Args:
            samples: List of column sample dicts from the generator
            tokenizer_name: HuggingFace tokenizer name
            max_seq_length: Maximum token sequence length
            label_encoder: Fitted LabelEncoder (if None, fits on this data)
            stat_means: Means for stat feature normalization
            stat_stds: Stds for stat feature normalization
        """
        self.samples = samples
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Encode labels
        labels = [s["label"] for s in samples]
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
        else:
            self.label_encoder = label_encoder
        self.labels = self.label_encoder.transform(labels)

        # Extract and normalize statistical features
        stat_matrix = np.array([
            [s["stats"][k] for k in self.STAT_KEYS] for s in samples
        ], dtype=np.float32)

        if stat_means is None or stat_stds is None:
            self.stat_means = stat_matrix.mean(axis=0)
            self.stat_stds = stat_matrix.std(axis=0) + 1e-8
        else:
            self.stat_means = stat_means
            self.stat_stds = stat_stds

        self.stat_features = (stat_matrix - self.stat_means) / self.stat_stds

        # Extract pattern features (already in [0, 1])
        self.pattern_features = np.array([
            [s["patterns"][k] for k in self.PATTERN_KEYS] for s in samples
        ], dtype=np.float32)

    def _build_text(self, sample: dict[str, Any]) -> str:
        """Build text input from header and values.

        Format: "header: <header> [SEP] values: <v1>, <v2>, ..."
        """
        header = sample["header"]
        values_str = ", ".join(v for v in sample["values"] if v.strip())
        return f"header: {header} [SEP] values: {values_str}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = self._build_text(sample)

        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "stat_features": torch.tensor(self.stat_features[idx], dtype=torch.float32),
            "pattern_features": torch.tensor(self.pattern_features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_dataloaders(
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    tokenizer_name: str = "distilbert-base-uncased",
    max_seq_length: int = 128,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, ColumnDataset]:
    """Create train, validation, and test DataLoaders.

    Normalization statistics are computed on training data and applied
    to validation and test sets to prevent data leakage.

    Args:
        train_samples: Training column samples
        val_samples: Validation column samples
        test_samples: Test column samples
        tokenizer_name: HuggingFace tokenizer name
        max_seq_length: Maximum sequence length
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset)
    """
    train_dataset = ColumnDataset(
        train_samples,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
    )

    val_dataset = ColumnDataset(
        val_samples,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        label_encoder=train_dataset.label_encoder,
        stat_means=train_dataset.stat_means,
        stat_stds=train_dataset.stat_stds,
    )

    test_dataset = ColumnDataset(
        test_samples,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        label_encoder=train_dataset.label_encoder,
        stat_means=train_dataset.stat_means,
        stat_stds=train_dataset.stat_stds,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset
