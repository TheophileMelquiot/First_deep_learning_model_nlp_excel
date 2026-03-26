"""Evaluation metrics for column classification.

Computes accuracy, macro F1-score, and ROC-AUC using scikit-learn.
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
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate model on a dataset and compute all metrics.

    Args:
        model: Trained column classifier
        data_loader: DataLoader for evaluation data
        device: Device to run inference on
        class_names: List of class label names

    Returns:
        Dictionary with metrics, predictions, and probabilities
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

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # ROC-AUC (one-vs-rest, macro)
    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }
