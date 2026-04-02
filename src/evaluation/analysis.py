"""Analysis and visualization tools for model evaluation.

Generates:
- Learning curves (train vs val loss/accuracy)
- Confusion matrix heatmap
- Error analysis with qualitative examples
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_learning_curves(
    history: dict,
    save_path: str = "outputs/learning_curves.png",
) -> None:
    """Plot training vs validation loss and accuracy curves.

    Args:
        history: Training history dict with train_loss, val_loss, etc.
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str = "outputs/confusion_matrix.png",
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array
        class_names: List of class label names
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_reliability_diagram(
    results: dict,
    save_path: str = "outputs/reliability_diagram.png",
) -> None:
    """Plot a reliability diagram (calibration plot).

    Shows whether predicted confidence matches actual accuracy.
    A perfectly calibrated model follows the diagonal y=x.
    Points ABOVE the diagonal → under-confident.
    Points BELOW the diagonal → over-confident (common with fine-tuned LLMs).

    Args:
        results: Output dict from evaluate_model() containing calibration_bins
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    bins = results["calibration_bins"]
    bin_confs = bins["confidences"]
    bin_accs = bins["accuracies"]
    bin_counts = bins["counts"]
    ece = results["ece"]

    # Only plot non-empty bins
    mask = bin_counts > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax1.bar(
        bin_confs[mask], bin_accs[mask],
        width=1.0 / len(bin_counts), alpha=0.6,
        color="steelblue", edgecolor="black", label="Model",
        align="center",
    )
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Correct Predictions")
    ax1.set_title(f"Reliability Diagram\nECE = {ece:.4f}")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right: Confidence histogram (how often each confidence level is used)
    ax2.bar(
        bin_confs[mask], bin_counts[mask],
        width=1.0 / len(bin_counts), alpha=0.7,
        color="coral", edgecolor="black", align="center",
    )
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Confidence Distribution\n(are predictions clustered near 1.0?)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved to {save_path}")


def plot_per_class_confidence(
    results: dict,
    save_path: str = "outputs/per_class_confidence.png",
) -> None:
    """Plot per-class mean confidence and entropy.

    Detects classes where the model is:
    - Over-confident (high confidence + possible errors)
    - Under-confident (low confidence = model unsure)
    - Confused (high entropy = probability mass spread across classes)

    Args:
        results: Output dict from evaluate_model()
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    per_class = results["per_class_confidence"]
    if not per_class:
        print("No per-class confidence data available.")
        return

    class_names = list(per_class.keys())
    confidences = [per_class[c]["mean_confidence"] for c in class_names]
    entropies = [per_class[c]["mean_entropy"] for c in class_names]
    f1_scores = [per_class[c]["f1"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # Top: Confidence vs F1 per class
    bars1 = ax1.bar(x - width / 2, confidences, width, label="Mean Confidence", color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, f1_scores, width, label="F1 Score", color="seagreen", alpha=0.8)
    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=35, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("Per-Class: Mean Confidence vs F1 Score\n(gap = over-confidence)")
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=7,
        )

    # Bottom: Mean prediction entropy per class
    ax2.bar(x, entropies, color="coral", alpha=0.8, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=35, ha="right")
    ax2.set_ylabel("Mean Prediction Entropy")
    ax2.set_title(
        f"Per-Class: Mean Prediction Entropy\n"
        f"(Overall: mean={results['mean_entropy']:.3f}, "
        f"normalized={results['mean_entropy_normalized']:.3f})"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class confidence plot saved to {save_path}")


def error_analysis(
    samples: list[dict],
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    num_examples: int = 10,
) -> list[dict]:
    """Analyze misclassified examples qualitatively.

    Args:
        samples: Original sample dictionaries
        predictions: Model predictions (integer labels)
        labels: True labels (integer labels)
        class_names: List of class label names
        num_examples: Number of error examples to return

    Returns:
        List of error analysis dictionaries
    """
    errors = []
    for i in range(len(predictions)):
        if predictions[i] != labels[i]:
            errors.append({
                "index": i,
                "header": samples[i]["header"],
                "values_sample": samples[i]["values"][:3],
                "true_label": class_names[labels[i]],
                "predicted_label": class_names[predictions[i]],
                "stats": samples[i]["stats"],
                "patterns": samples[i]["patterns"],
            })

    # Return a diverse sample of errors
    if len(errors) > num_examples:
        step = len(errors) // num_examples
        errors = errors[::step][:num_examples]

    return errors


def print_error_analysis(errors: list[dict]) -> None:
    """Print error analysis in a readable format.

    Args:
        errors: List of error dictionaries from error_analysis()
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS - Misclassified Examples")
    print("=" * 60)

    for i, err in enumerate(errors, 1):
        print(f"\n--- Error {i} ---")
        print(f"  Header:    {err['header']}")
        print(f"  Values:    {err['values_sample']}")
        print(f"  True:      {err['true_label']}")
        print(f"  Predicted: {err['predicted_label']}")
        print(f"  Stats:     {err['stats']}")
        print(f"  Patterns:  {err['patterns']}")
