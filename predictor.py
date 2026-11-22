"""Loads the trained artifacts once and turns review text into a label + probability."""
import json
import re
from pathlib import Path

import torch

from model import GRUModel

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def tokenize(text: str) -> list[str]:
    """Exactly the notebook's tokenizer: lowercase, strip HTML, keep "don't" as one token."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


class SentimentPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        with open(artifacts_dir / "config.json") as f:
            self.config = json.load(f)
        with open(artifacts_dir / "word2idx.json") as f:
            self.word2idx = json.load(f)

        self.max_len = self.config["max_len"]
        self.model = GRUModel(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
        )
        self.model.load_state_dict(
            torch.load(artifacts_dir / "gru_model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def encode(self, text: str) -> tuple[list[int], int]:
        """Same as the notebook's encode_review: truncate, map unknowns to 1, pad with 0."""
        words = tokenize(text)[: self.max_len]
        ids = [self.word2idx.get(w, 1) for w in words]
        length = len(ids)
        ids = ids + [0] * (self.max_len - length)
        return ids, length

    def predict(self, text: str) -> dict:
        ids, length = self.encode(text)
        if length == 0:
            # No usable words (e.g. only digits/symbols); the model would index position -1.
            raise ValueError("Review contains no recognisable words.")
        ids_t = torch.tensor([ids], dtype=torch.long)
        lengths_t = torch.tensor([length], dtype=torch.long)
        with torch.inference_mode():
            logit = self.model(ids_t, lengths_t)[0]
            p_positive = torch.sigmoid(logit).item()   # model was trained to output P(positive)

        if p_positive >= 0.5:
            label, confidence = "POSITIVE", p_positive
        else:
            label, confidence = "NEGATIVE", 1.0 - p_positive
        return {"label": label, "probability": round(confidence, 4)}
