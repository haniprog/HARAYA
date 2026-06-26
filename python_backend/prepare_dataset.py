"""
Dataset preparation script to convert combined_dataset.csv 
into the format required by train.py

Layer 2: Data Preparation - Baguhin ang format ng combined dataset
para sa BERT training.
"""
from pathlib import Path
import pandas as pd


def prepare_combined_dataset(input_csv: Path, output_csv: Path) -> None:
    """
    Converts combined_dataset.csv to the training format.
    
    Input columns: cleaned_text, tokens, label
    Output columns: conversation, label
    
    The label values will be normalized to lowercase for consistency.
    """
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Baguhin ang column name para tumugma sa training script expectations.
    df = df.rename(columns={"cleaned_text": "conversation"})
    
    # Normalize labels to lowercase para sa consistency sa LABEL_TO_ID mapping.
    df["label"] = df["label"].str.lower()
    
    # Kunin lang ang columns na kailangan natin.
    df = df[["conversation", "label"]]
    
    # Alisin ang rows na may missing values dahil hindi sila makakagana sa training.
    df = df.dropna()
    
    print(f"✓ Inihanda ang dataset na may {len(df)} samples")
    print(f"📊 Distribution ng labels:\n{df['label'].value_counts()}")
    
    # Lumikha ng output directory kung hindi pa nandoon.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # I-save ang processed dataset sa CSV format.
    df.to_csv(output_csv, index=False)
    print(f"✓ Dataset saved to {output_csv}")


if __name__ == "__main__":
    # Paths
    input_dataset = Path(__file__).resolve().parents[1] / "combined_dataset.csv"
    output_dataset = Path(__file__).resolve().parent / "data" / "training_dataset.csv"
    
    if not input_dataset.exists():
        print(f"Error: {input_dataset} not found")
        exit(1)
    
    prepare_combined_dataset(input_dataset, output_dataset)
