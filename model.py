"""GRU sentiment classifier -- identical architecture to Class 23 (SentimentClassifier with nn.GRU)."""
import torch
from torch import nn


class GRUModel(nn.Module):
    """Word IDs -> Embedding -> GRU -> last real-word output -> Linear -> one logit."""

    def __init__(self, vocab_size: int, embed_dim: int = 100,
                 hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Attribute is named `rnn` (not `gru`) so the saved state_dict keys match the notebook.
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(ids)                       # (batch, max_len, embed_dim)
        outputs, _ = self.rnn(vectors)                      # (batch, max_len, hidden)
        rows = torch.arange(ids.size(0), device=ids.device)
        last_real = outputs[rows, lengths - 1]              # output after each review's last real word
        return self.fc(last_real).squeeze(1)                # (batch,) raw logits
