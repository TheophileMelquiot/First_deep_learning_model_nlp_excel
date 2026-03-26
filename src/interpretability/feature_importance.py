"""Interpretability tools for the column classifier.

Provides feature importance analysis to understand which modalities
(text, statistics, patterns) contribute most to predictions.
Uses a permutation-based approach and attention weight analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


@torch.no_grad()
def permutation_feature_importance(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    n_repeats: int = 5,
) -> dict[str, float]:
    """Compute feature importance via permutation.

    Measures the drop in accuracy when each feature group is
    randomly shuffled, indicating its importance.

    Args:
        model: Trained model
        data_loader: Evaluation data loader
        device: Device for inference
        n_repeats: Number of permutation repeats

    Returns:
        Dictionary mapping feature group names to importance scores
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Collect all data
    all_input_ids = []
    all_attention_mask = []
    all_stat_features = []
    all_pattern_features = []
    all_labels = []

    for batch in data_loader:
        all_input_ids.append(batch["input_ids"])
        all_attention_mask.append(batch["attention_mask"])
        all_stat_features.append(batch["stat_features"])
        all_pattern_features.append(batch["pattern_features"])
        all_labels.append(batch["label"])

    input_ids = torch.cat(all_input_ids).to(device)
    attention_mask = torch.cat(all_attention_mask).to(device)
    stat_features = torch.cat(all_stat_features).to(device)
    pattern_features = torch.cat(all_pattern_features).to(device)
    labels = torch.cat(all_labels).to(device)

    # Baseline accuracy
    logits = model(input_ids, attention_mask, stat_features, pattern_features)
    baseline_acc = (logits.argmax(1) == labels).float().mean().item()

    importance = {}
    feature_groups = {
        "stat_features": stat_features,
        "pattern_features": pattern_features,
    }

    for name, features in feature_groups.items():
        drops = []
        for _ in range(n_repeats):
            # Shuffle this feature group
            perm_idx = torch.randperm(features.size(0), device=device)
            shuffled = features[perm_idx]

            # Forward pass with shuffled features
            if name == "stat_features":
                logits = model(input_ids, attention_mask, shuffled, pattern_features)
            else:
                logits = model(input_ids, attention_mask, stat_features, shuffled)

            perm_acc = (logits.argmax(1) == labels).float().mean().item()
            drops.append(baseline_acc - perm_acc)

        importance[name] = float(np.mean(drops))

    return importance, baseline_acc


def plot_feature_importance(
    importance: dict[str, float],
    baseline_acc: float,
    save_path: str = "outputs/feature_importance.png",
) -> None:
    """Plot feature importance as a bar chart.

    Args:
        importance: Feature importance scores
        baseline_acc: Baseline accuracy for reference
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    names = list(importance.keys())
    values = list(importance.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if v > 0 else "#FF5722" for v in values]
    bars = ax.bar(names, values, color=colors, edgecolor="black", alpha=0.8)

    ax.set_ylabel("Accuracy Drop (higher = more important)")
    ax.set_title(f"Feature Importance (Baseline Acc: {baseline_acc:.4f})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {save_path}")
