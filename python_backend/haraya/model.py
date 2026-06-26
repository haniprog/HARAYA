from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast

from .config import CONFIG


# Layer 5: Improved BERT Model
# Dito ang BERT backbone at classification head na tinetrain o ginagamit sa inference.
class HarayaBertClassifier(nn.Module):
    def __init__(self, model_name: str, num_behavioral_features: int, num_labels: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + num_behavioral_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, behavioral_features):
        # Kunin ang contextual embedding mula sa BERT.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        # I-fuse ang BERT embedding at behavioral features.
        combined = torch.cat([pooled, behavioral_features], dim=1)
        return self.classifier(combined)


def load_tokenizer(model_name: str = CONFIG.model_name) -> BertTokenizerFast:
    # Tokenizer na naghahati ng text sa mga token na alam ng BERT.
    return BertTokenizerFast.from_pretrained(model_name)


def load_model(
    model_name: str = CONFIG.model_name,
    weights_path: Optional[Path] = None,
) -> HarayaBertClassifier:
    # Gumawa ng model at mag-load ng trained weights kung meron.
    model = HarayaBertClassifier(
        model_name=model_name,
        num_behavioral_features=CONFIG.num_behavioral_features,
        num_labels=CONFIG.num_labels,
    )

    if weights_path is not None and weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.has_trained_weights = True
    else:
        model.has_trained_weights = False

    model.eval()
    return model


def softmax_probabilities(logits: torch.Tensor) -> np.ndarray:
    # Gawing probability ang raw logits para mabasa ang confidence ng classes.
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities.detach().cpu().numpy()[0]
