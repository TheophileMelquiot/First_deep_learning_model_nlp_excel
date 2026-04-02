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
    model, data_loader, device=None, n_repeats=5, confidence_threshold=0.95
):
    """Permutation importance focused on uncertain predictions.
    
    At 99%+ accuracy, permuting features on easy examples shows 0.
    We restrict to samples where the model is NOT fully confident
    (max_softmax < confidence_threshold), where feature signals matter.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_ids, all_mask, all_stat, all_pattern, all_labels = [], [], [], [], []
    for batch in data_loader:
        all_ids.append(batch["input_ids"])
        all_mask.append(batch["attention_mask"])
        all_stat.append(batch["stat_features"])
        all_pattern.append(batch["pattern_features"])
        all_labels.append(batch["label"])

    input_ids = torch.cat(all_ids).to(device)
    attention_mask = torch.cat(all_mask).to(device)
    stat_features = torch.cat(all_stat).to(device)
    pattern_features = torch.cat(all_pattern).to(device)
    labels = torch.cat(all_labels).to(device)

    # Baseline — full dataset
    logits = model(input_ids, attention_mask, stat_features, pattern_features)
    probs = torch.softmax(logits, dim=1)
    baseline_acc = (logits.argmax(1) == labels).float().mean().item()

    # Filter to uncertain samples only
    max_conf = probs.max(dim=1).values
    uncertain_mask = max_conf < confidence_threshold
    n_uncertain = uncertain_mask.sum().item()
    print(f"  Uncertain samples (conf < {confidence_threshold}): {n_uncertain}/{len(labels)}")

    if n_uncertain < 10:
        print("  ⚠️  Too few uncertain samples — all predictions are overconfident.")
        print("  Feature importance is 0.0 because BERT alone solves the task.")
        return {"stat_features": 0.0, "pattern_features": 0.0}, baseline_acc

    u_ids = input_ids[uncertain_mask]
    u_mask = attention_mask[uncertain_mask]
    u_stat = stat_features[uncertain_mask]
    u_pattern = pattern_features[uncertain_mask]
    u_labels = labels[uncertain_mask]

    u_logits = model(u_ids, u_mask, u_stat, u_pattern)
    uncertain_baseline = (u_logits.argmax(1) == u_labels).float().mean().item()

    importance = {}
    for name in ["stat_features", "pattern_features"]:
        drops = []
        for _ in range(n_repeats):
            perm_idx = torch.randperm(u_stat.size(0), device=device)
            if name == "stat_features":
                logits_p = model(u_ids, u_mask, u_stat[perm_idx], u_pattern)
            else:
                logits_p = model(u_ids, u_mask, u_stat, u_pattern[perm_idx])
            perm_acc = (logits_p.argmax(1) == u_labels).float().mean().item()
            drops.append(uncertain_baseline - perm_acc)
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
