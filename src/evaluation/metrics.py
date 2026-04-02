"""Evaluation metrics for column classification.

Computes accuracy, macro F1-score, ROC-AUC, calibration (ECE, Brier),
per-class confidence, and prediction entropy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted confidence aligns with actual accuracy.
    A well-calibrated model has ECE close to 0.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples,)
        n_bins: Number of confidence bins

    Returns:
        Tuple of (ece, bin_confidences, bin_accuracies, bin_counts)
    """
    confidences = probs.max(axis=1)          # max prob = model confidence
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidences[i] = confidences[mask].mean()
            bin_accuracies[i] = correct[mask].mean()
            bin_counts[i] = mask.sum()

    # ECE = weighted average of |confidence - accuracy| per bin
    ece = np.sum(
        bin_counts / len(labels) * np.abs(bin_confidences - bin_accuracies)
    )
    return ece, bin_confidences, bin_accuracies, bin_counts


def _prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-sample prediction entropy.

    High entropy = uncertain prediction.
    Low entropy  = confident prediction (can be over-confident if wrong).

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        Entropy per sample, shape (n_samples,)
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    return -(probs_clipped * np.log(probs_clipped)).sum(axis=1)


def _overfitting_metrics(
    train_loss: float | None,
    val_loss: float | None,
    train_acc: float | None,
    val_acc: float | None,
) -> dict:
    """Compute train/val gap metrics for overfitting diagnosis.

    Args:
        train_loss: Final training loss
        val_loss: Final validation loss
        train_acc: Final training accuracy
        val_acc: Final validation accuracy

    Returns:
        Dictionary with gap metrics
    """
    metrics = {}
    if train_loss is not None and val_loss is not None:
        metrics["loss_gap"] = val_loss - train_loss      # positive = val worse (normal)
        metrics["loss_ratio"] = val_loss / (train_loss + 1e-10)
    if train_acc is not None and val_acc is not None:
        metrics["acc_gap"] = train_acc - val_acc         # positive = train better (overfit)
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    class_names: list[str] | None = None,
    train_loss: float | None = None,
    val_loss: float | None = None,
    train_acc: float | None = None,
    val_acc: float | None = None,
) -> dict:
    """Evaluate model and compute comprehensive metrics.

    New metrics vs original:
    - ECE (Expected Calibration Error): is the model's confidence realistic?
    - Brier Score: probabilistic accuracy (lower = better)
    - Log Loss (Cross-Entropy on test): how uncertain are predictions?
    - Mean/Std prediction entropy: detects over-confident or collapsed predictions
    - Per-class confidence: finds which classes the model is uncertain about
    - Overfitting gaps: train vs val loss/accuracy delta

    Args:
        model: Trained column classifier
        data_loader: DataLoader for evaluation data
        device: Device to run inference on
        class_names: List of class label names
        train_loss: Final train loss from history (for gap metrics)
        val_loss: Final val loss from history (for gap metrics)
        train_acc: Final train accuracy from history
        val_acc: Final val accuracy from history

    Returns:
        Dictionary with all metrics, predictions, and probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        stat_features = batch["stat_features"].to(device)
        pattern_features = batch["pattern_features"].to(device)
        labels = batch["label"]

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stat_features=stat_features,
            pattern_features=pattern_features,
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    n_classes = all_probs.shape[1]

    # ── Original metrics ──────────────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    ece, bin_confs, bin_accs, bin_counts = _expected_calibration_error(
        all_probs, all_labels
    )

    # Brier score (one-vs-rest average across classes)
    one_hot = np.zeros_like(all_probs)
    one_hot[np.arange(len(all_labels)), all_labels] = 1.0
    brier = np.mean((all_probs - one_hot) ** 2)

    # Log loss on test set (== cross-entropy, comparable to train/val loss)
    test_log_loss = log_loss(all_labels, all_probs)

    # ── Uncertainty / entropy ─────────────────────────────────────────────────
    entropy = _prediction_entropy(all_probs)
    mean_entropy = float(entropy.mean())
    std_entropy = float(entropy.std())
    # Max possible entropy for n_classes (uniform distribution)
    max_entropy = float(np.log(n_classes))
    # Normalized: 0 = fully confident, 1 = fully uncertain
    mean_entropy_normalized = mean_entropy / max_entropy

    # ── Per-class confidence ──────────────────────────────────────────────────
    per_class_confidence = {}
    if class_names is not None:
        for class_idx, name in enumerate(class_names):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                per_class_confidence[name] = {
                    "mean_confidence": float(all_probs[mask, class_idx].mean()),
                    "mean_entropy": float(entropy[mask].mean()),
                    "f1": float(f1_per_class[class_idx]) if class_idx < len(f1_per_class) else float("nan"),
                }

    # ── Overfitting gaps ──────────────────────────────────────────────────────
    overfitting = _overfitting_metrics(train_loss, val_loss, train_acc, val_acc)
    # Add test vs val gap if val_loss provided
    if val_loss is not None:
        overfitting["test_log_loss_vs_val_loss"] = test_log_loss - val_loss

    return {
        # Original
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        # Calibration
        "ece": ece,
        "brier_score": brier,
        "test_log_loss": test_log_loss,
        "calibration_bins": {
            "confidences": bin_confs,
            "accuracies": bin_accs,
            "counts": bin_counts,
        },
        # Uncertainty
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "mean_entropy_normalized": mean_entropy_normalized,
        # Per-class
        "per_class_confidence": per_class_confidence,
        # Overfitting
        "overfitting_gaps": overfitting,
    }
