from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from haraya.config import CONFIG
from haraya.features import build_behavioral_features, to_feature_vector
from haraya.model import HarayaBertClassifier, load_tokenizer
from haraya.preprocessing import preprocess_conversation


# Training Script: Dataset -> Preprocess -> BERT -> Behavioral Features -> Classification
# Dito dumadaan ang pag-train ng model mula CSV hanggang saved weights.
LABEL_TO_ID = {
    "safe": 0,
    "potential": 1,
    "harassment": 2,
}


class HarayaDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        # Kinukuha ang isang row at ginagawang input na kaya basahin ng BERT.
        row = self.dataframe.iloc[index]
        messages = [segment.strip() for segment in str(row["conversation"]).split("[SEP]")]
        preprocessed = preprocess_conversation(messages)
        behavioral = build_behavioral_features(messages)
        tokens = self.tokenizer(
            preprocessed.context or " ",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=CONFIG.max_length,
        )

        label = LABEL_TO_ID[str(row["label"]).strip().lower()]
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "behavioral_features": torch.tensor(to_feature_vector(behavioral), dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_batch(batch):
    # Pinagsasama ang maraming sample para sabay-sabay i-train sa isang batch.
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "behavioral_features": torch.stack([item["behavioral_features"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def train_model(dataset_path: Path, output_path: Path, epochs: int = 1, batch_size: int = 4) -> None:
    # Basahin ang CSV dataset at hatiin sa train/validation.
    dataframe = pd.read_csv(dataset_path)
    train_frame, valid_frame = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label"],
    )

    tokenizer = load_tokenizer()
    train_dataset = HarayaDataset(train_frame, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    model = HarayaBertClassifier(
        model_name=CONFIG.model_name,
        num_behavioral_features=CONFIG.num_behavioral_features,
        num_labels=CONFIG.num_labels,
    )
    # Optimizer at loss function para matuto ang model.
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            # Forward pass -> loss -> backpropagation -> update weights.
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                behavioral_features=batch["behavioral_features"],
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()

    model.eval()
    # I-save ang natutunan na weights para magamit ng API sa inference.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def main() -> None:
    # Command-line entrypoint para madaling patakbuhin ang training script.
    parser = argparse.ArgumentParser(description="Train the HARAYA BERT classifier")
    parser.add_argument("--dataset", required=True, type=Path, help="CSV file with conversation and label columns")
    parser.add_argument("--output", default=Path("python_backend/artifacts/haraya_bert.pt"), type=Path)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    train_model(args.dataset, args.output, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()