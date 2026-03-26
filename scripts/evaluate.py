"""Standalone evaluation script.

Loads a trained model and evaluates it on test data.

Usage:
    python -m scripts.evaluate --model outputs/model.pt [--config config/config.yaml]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.generator import generate_dataset
from src.data.dataset import create_dataloaders
from src.model.classifier import ColumnClassifier
from src.evaluation.metrics import evaluate_model
from src.evaluation.analysis import (
    plot_confusion_matrix,
    error_analysis,
    print_error_analysis,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Column Classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default="outputs/model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    # Regenerate dataset (same seed ensures same split)
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

    _, _, test_loader, train_dataset = create_dataloaders(
        train_samples, val_samples, test_samples,
        tokenizer_name=config["model"]["transformer_name"],
        max_seq_length=config["training"]["max_seq_length"],
        batch_size=config["training"]["batch_size"],
    )
    class_names = list(train_dataset.label_encoder.classes_)

    # Load model
    model = ColumnClassifier(
        num_classes=len(class_names),
        transformer_name=config["model"]["transformer_name"],
        num_stat_features=config["model"]["num_stat_features"],
        num_pattern_features=config["model"]["num_pattern_features"],
        feature_hidden_size=config["model"]["feature_hidden_size"],
        fusion_hidden_size=config["model"]["fusion_hidden_size"],
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(torch.load(args.model, weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    results = evaluate_model(model, test_loader, device=device, class_names=class_names)

    print(f"Test Accuracy:  {results['accuracy']:.4f}")
    print(f"Test F1 (macro): {results['f1_macro']:.4f}")
    print(f"Test ROC-AUC:   {results['roc_auc']:.4f}")

    # Confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"], class_names,
        save_path="outputs/confusion_matrix_eval.png",
    )

    # Error analysis
    errors = error_analysis(
        test_samples, results["predictions"], results["labels"],
        class_names, num_examples=10,
    )
    print_error_analysis(errors)


if __name__ == "__main__":
    main()
