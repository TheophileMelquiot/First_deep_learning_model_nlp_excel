"""Main training script for the column classifier.

Usage:
    python -m scripts.train [--config config/config.yaml]
"""

import argparse
import random
import sys
from pathlib import Path
import json 
import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.generator import generate_dataset
from src.data.dataset import create_dataloaders
from src.model.classifier import ColumnClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.evaluation.analysis import (
    plot_learning_curves,
    plot_confusion_matrix,
    error_analysis,
    print_error_analysis,
)
from src.interpretability.feature_importance import (
    permutation_feature_importance,
    plot_feature_importance,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def split_dataset(
    dataset: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list, list, list]:
    """Split dataset into train, validation, and test sets."""
    n = len(dataset)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_samples = dataset[:n_test]
    val_samples = dataset[n_test:n_test + n_val]
    train_samples = dataset[n_test + n_val:]

    return train_samples, val_samples, test_samples

def print_overfitting_report(results: dict, history: dict) -> None:
    """Print a structured overfitting diagnostic report."""
    print("\n" + "=" * 60)
    print("OVERFITTING / CALIBRATION DIAGNOSTIC REPORT")
    print("=" * 60)

    # ── Calibration metrics ───────────────────────────────────────
    print("\n📐 Calibration:")
    ece = results["ece"]
    brier = results["brier_score"]
    test_ll = results["test_log_loss"]
    print(f"  ECE (Expected Calibration Error) : {ece:.4f}  "
          f"{'✅ good (<0.05)' if ece < 0.05 else '⚠️  check reliability diagram'}")
    print(f"  Brier Score                      : {brier:.4f}  "
          f"{'✅ good (<0.05)' if brier < 0.05 else '⚠️  probabilistic error is high'}")
    print(f"  Test Log Loss                    : {test_ll:.4f}")

    # ── Uncertainty / entropy ─────────────────────────────────────
    print("\n🔮 Prediction Uncertainty:")
    me = results["mean_entropy"]
    me_norm = results["mean_entropy_normalized"]
    print(f"  Mean prediction entropy          : {me:.4f}")
    print(f"  Normalized entropy (0=sure,1=lost): {me_norm:.4f}  "
          f"{'⚠️  suspiciously low' if me_norm < 0.01 else '✅ ok'}")

    # ── Overfitting gaps ──────────────────────────────────────────
    gaps = results["overfitting_gaps"]
    if gaps:
        print("\n📊 Train / Val / Test Gaps:")
        if "loss_gap" in gaps:
            g = gaps["loss_gap"]
            print(f"  Val Loss - Train Loss            : {g:+.4f}  "
                  f"{'✅ normal' if g > 0 else '⚠️  val < train (data leak?)'}")
        if "acc_gap" in gaps:
            g = gaps["acc_gap"]
            print(f"  Train Acc - Val Acc              : {g:+.4f}  "
                  f"{'⚠️  overfitting' if g > 0.05 else '✅ ok'}")
        if "test_log_loss_vs_val_loss" in gaps:
            g = gaps["test_log_loss_vs_val_loss"]
            print(f"  Test LogLoss - Val Loss          : {g:+.4f}  "
                  f"{'✅ generalizing' if abs(g) < 0.1 else '⚠️  test/val distribution shift'}")

    # ── Per-class confidence ──────────────────────────────────────
    print("\n🔬 Per-Class Confidence vs F1 (potential over-confidence):")
    per_class = results["per_class_confidence"]
    header = f"  {'Class':<18} {'Confidence':>12} {'F1':>8} {'Entropy':>10} {'Status':>10}"
    print(header)
    print("  " + "-" * 60)
    for cls, m in per_class.items():
        conf = m["mean_confidence"]
        f1 = m["f1"]
        ent = m["mean_entropy"]
        # Flag over-confidence: confidence much higher than F1
        status = "⚠️  over-conf" if (conf - f1) > 0.05 else "✅"
        print(f"  {cls:<18} {conf:>12.4f} {f1:>8.4f} {ent:>10.4f} {status:>10}")

    print()

def main():
    parser = argparse.ArgumentParser(description="Train Column Classifier")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config["seed"])

    print("=" * 60)
    print("COLUMN CLASSIFICATION - Training Pipeline")
    print("=" * 60)

    # Generate dataset
    print("\n[1/5] Loading merged dataset...")
    with open("data/merged_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)
    random.shuffle(dataset)   # seed already set above → reproducible
    print(f"  Total samples: {len(dataset)}")

    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        dataset,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Create data loaders
    print("\n[2/5] Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        train_samples, val_samples, test_samples,
        tokenizer_name=config["model"]["transformer_name"],
        max_seq_length=config["training"]["max_seq_length"],
        batch_size=config["training"]["batch_size"],
    )
    class_names = list(train_dataset.label_encoder.classes_)
    print(f"  Classes: {class_names}")

    # Build model
    print("\n[3/5] Building model...")
    model = ColumnClassifier(
        num_classes=len(class_names),
        transformer_name=config["model"]["transformer_name"],
        num_stat_features=config["model"]["num_stat_features"],
        num_pattern_features=config["model"]["num_pattern_features"],
        feature_hidden_size=config["model"]["feature_hidden_size"],
        fusion_hidden_size=config["model"]["fusion_hidden_size"],
        dropout=config["model"]["dropout"],
        freeze_transformer=config["model"]["freeze_transformer"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    print("\n[4/5] Training model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
        min_delta=config["training"]["min_delta"],
    )
    history = trainer.train()

    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    device = trainer.device
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)

    print(f"\n  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test F1 (macro): {results['f1_macro']:.4f}")
    print(f"  Test ROC-AUC: {results['roc_auc']:.4f}")

    # Generate analysis outputs
    print("\nGenerating analysis outputs...")

    # Learning curves
    plot_learning_curves(history, save_path="outputs/learning_curves.png")

    # Confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        save_path="outputs/confusion_matrix.png",
    )

    # Error analysis
    errors = error_analysis(
        test_samples, results["predictions"], results["labels"],
        class_names, num_examples=10,
    )
    print_error_analysis(errors)

    # Feature importance
    print("\nComputing feature importance...")
    importance, baseline_acc = permutation_feature_importance(
        model, test_loader, device=device, n_repeats=3,
    )
    plot_feature_importance(importance, baseline_acc, save_path="outputs/feature_importance.png")

    # Save model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(f"\nModel saved to {output_dir / 'model.pt'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
