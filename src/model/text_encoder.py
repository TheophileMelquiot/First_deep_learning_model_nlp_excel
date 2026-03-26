"""Text encoder using a pretrained DistilBERT transformer.

Encodes the concatenated header + values text into a fixed-size
embedding using the [CLS] token representation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Encode text (header + values) using a pretrained transformer.

    Architecture:
        Input text → DistilBERT → [CLS] token embedding (768-dim)

    The [CLS] token captures a summary representation of the full
    input sequence, making it suitable for classification tasks.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze: bool = False,
    ):
        """Initialize the text encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze transformer weights
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the transformer output."""
        return self.transformer.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_length)
            attention_mask: Attention mask, shape (batch_size, seq_length)

        Returns:
            [CLS] token embeddings, shape (batch_size, hidden_size)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token (first token) as sequence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
