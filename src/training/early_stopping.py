"""Early stopping mechanism for training.

Monitors validation loss and stops training when no improvement
is observed for a specified number of epochs (patience).
"""


class EarlyStopping:
    """Early stopping to prevent overfitting.

    Tracks validation loss and signals when training should stop
    if no improvement is seen for `patience` consecutive epochs.
    Also saves the best model state.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: PyTorch model (for saving best state)

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _save_checkpoint(self, model) -> None:
        """Save model state dict as best checkpoint."""
        import copy
        self.best_model_state = copy.deepcopy(model.state_dict())
