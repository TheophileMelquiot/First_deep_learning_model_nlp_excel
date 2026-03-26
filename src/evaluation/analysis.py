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
