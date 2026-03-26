"""Ablation study script.

Trains and evaluates model variants with different modality combinations
to understand the contribution of each input modality.

Ablation configurations:
1. Full model (all modalities)
2. Without header text
3. Without value text
4. Without statistical features
5. Without pattern features
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.generator import generate_dataset
from src.data.dataset import ColumnDataset, create_dataloaders
from src.model.classifier import ColumnClassifier
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ABLATION_CONFIGS = {
    "full_model": {
        "use_header": True,
        "use_values": True,
        "use_stats": True,
        "use_patterns": True,
    },
    "no_header": {
        "use_header": False,
        "use_values": True,
        "use_stats": True,
        "use_patterns": True,
    },
    "no_values": {
        "use_header": True,
        "use_values": False,
        "use_stats": True,
        "use_patterns": True,
    },
    "no_stats": {
        "use_header": True,
        "use_values": True,
        "use_stats": False,
        "use_patterns": True,
    },
    "no_patterns": {
        "use_header": True,
        "use_values": True,
        "use_stats": True,
        "use_patterns": False,
    },
}


def run_ablation(config: dict, ablation_name: str, ablation_flags: dict,
                 train_samples, val_samples, test_samples, class_names) -> dict:
    """Run a single ablation experiment.

    Args:
        config: Full config dictionary
        ablation_name: Name of this ablation experiment
        ablation_flags: Dict of use_header, use_values, etc.
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        class_names: Class label names

    Returns:
        Results dictionary with metrics
    """
    print(f"\n{'=' * 50}")
    print(f"ABLATION: {ablation_name}")
    print(f"Config: {ablation_flags}")
    print(f"{'=' * 50}")

    set_seed(config["seed"])

    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        train_samples, val_samples, test_samples,
        tokenizer_name=config["model"]["transformer_name"],
        max_seq_length=config["training"]["max_seq_length"],
        batch_size=config["training"]["batch_size"],
    )

    # Build model with ablation flags
    model = ColumnClassifier(
        num_classes=len(class_names),
        transformer_name=config["model"]["transformer_name"],
        num_stat_features=config["model"]["num_stat_features"],
        num_pattern_features=config["model"]["num_pattern_features"],
        feature_hidden_size=config["model"]["feature_hidden_size"],
        fusion_hidden_size=config["model"]["fusion_hidden_size"],
        dropout=config["model"]["dropout"],
        freeze_transformer=config["model"]["freeze_transformer"],
        **ablation_flags,
    )

    # Train
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
    device = trainer.device
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)

    return {
        "name": ablation_name,
        "accuracy": results["accuracy"],
        "f1_macro": results["f1_macro"],
        "roc_auc": results["roc_auc"],
        "config": ablation_flags,
    }


def plot_ablation_results(
    results: list[dict],
    save_path: str = "outputs/ablation_results.png",
) -> None:
    """Plot ablation study results as grouped bar chart.

    Args:
        results: List of ablation result dictionaries
        save_path: Path to save the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    f1 = [r["f1_macro"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, f1, width, label="F1 (macro)", color="#FF9800", alpha=0.8)

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Impact of Each Modality")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation results plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(
        num_samples_per_class=config["data"]["num_samples_per_class"],
        class_labels=config["data"]["class_labels"],
        num_values=config["data"]["max_values_per_column"],
        seed=config["seed"],
    )

    n = len(dataset)
    n_test = int(n * config["data"]["test_ratio"])
    n_val = int(n * config["data"]["val_ratio"])
    test_samples = dataset[:n_test]
    val_samples = dataset[n_test:n_test + n_val]
    train_samples = dataset[n_test + n_val:]
    class_names = config["data"]["class_labels"]

    # Run ablation experiments
    all_results = []
    for name, flags in ABLATION_CONFIGS.items():
        result = run_ablation(
            config, name, flags,
            train_samples, val_samples, test_samples, class_names,
        )
        all_results.append(result)
        print(f"\n  {name}: Acc={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Convert results for JSON serialization
    json_results = []
    for r in all_results:
        json_results.append({
            "name": r["name"],
            "accuracy": float(r["accuracy"]),
            "f1_macro": float(r["f1_macro"]),
            "roc_auc": float(r["roc_auc"]),
            "config": r["config"],
        })

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Plot results
    plot_ablation_results(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['name']:20s} | Acc: {r['accuracy']:.4f} | F1: {r['f1_macro']:.4f} | AUC: {r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
