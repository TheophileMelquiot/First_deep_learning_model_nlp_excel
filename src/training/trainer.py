"""Training loop for the column classifier.

Implements a manual PyTorch training loop with:
- Cross-entropy loss
- AdamW optimizer
- Learning rate scheduling
- Early stopping
- Epoch-level logging of train/val metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping


class Trainer:
    """Manages the training and validation loop.

    Tracks training history (losses, metrics) for later analysis
    including learning curves and overfitting diagnostics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 20,
        patience: int = 5,
        min_delta: float = 0.001,
        device: str | None = None,
    ):
        """Initialize the trainer.

        Args:
            model: The column classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for AdamW
            weight_decay: Weight decay for regularization
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            min_delta: Early stopping minimum delta
            device: Device to use (auto-detected if None)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (AdamW with weight decay for regularization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            stat_features = batch["stat_features"].to(self.device)
            pattern_features = batch["pattern_features"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stat_features=stat_features,
                pattern_features=pattern_features,
            )
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        """Run validation.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            stat_features = batch["stat_features"].to(self.device)
            pattern_features = batch["pattern_features"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stat_features=stat_features,
                pattern_features=pattern_features,
            )
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Training history dictionary
        """
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print("-" * 60)

        for epoch in range(1, self.num_epochs + 1):
            # Training
            train_loss, train_acc = self._train_epoch()

            # Validation
            val_loss, val_acc = self._validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {lr:.2e}"
            )

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Restore best model
        if self.early_stopping.best_model_state is not None:
            self.model.load_state_dict(self.early_stopping.best_model_state)
            print("Restored best model weights")

        return self.history
