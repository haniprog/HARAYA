"""
Combined training script for HARAYA BERT classifier.
Prepares the combined dataset and trains the model in one go.

Layer 1 + 2: Data Input + Preparation
Dinadalang ang combined dataset at ginagawa itong training-ready.

Usage:
    python train_with_combined.py [--epochs 3] [--batch-size 8]
"""
from pathlib import Path
import pandas as pd
import argparse

from haraya.config import CONFIG
from train import train_model


def prepare_training_dataset(combined_csv: Path, output_csv: Path) -> None:
    """
    Converts combined_dataset.csv to the format expected by train.py:
    - Renames 'cleaned_text' column to 'conversation'
    - Normalizes labels to lowercase (safe interaction, potential, harassment)
    """
    print(f"📦 Loading combined dataset from {combined_csv}...")
    df = pd.read_csv(combined_csv)
    
    # Baguhin ang column name mula 'cleaned_text' tungo 'conversation'.
    df = df.rename(columns={"cleaned_text": "conversation"})
    
    # I-normalize ang labels sa lowercase para sa consistency sa training.
    df["label"] = df["label"].str.lower()
    
    # I-map ang label names para tumugma sa BERT training expectations.
    # Safe Interaction -> safe, Potential Harassment -> potential, Harassment -> harassment
    label_mapping = {
        "safe interaction": "safe",
        "potential harassment": "potential", 
        "harassment": "harassment"
    }
    df["label"] = df["label"].map(label_mapping)
    
    # Kunin lang ang kailangan na columns at alisin ang null values.
    df = df[["conversation", "label"]].dropna()
    
    print(f"✓ Inihanda ang {len(df)} samples")
    print(f"📊 Distribution ng labels:\n{df['label'].value_counts()}\n")
    
    # Lumikha ng directory structure para sa training data.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # I-save ang prepared dataset bilang CSV.
    df.to_csv(output_csv, index=False)
    print(f"✓ Training dataset saved to {output_csv}")
    
    return output_csv


def main():
    # Command-line entrypoint para madaling patakbuhin ang full training pipeline.
    parser = argparse.ArgumentParser(
        description="Train HARAYA BERT classifier with combined dataset"
    )
    parser.add_argument("--epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--combined-dataset", default="../combined_dataset.csv", 
                        type=Path, help="Path to combined_dataset.csv")
    parser.add_argument("--output", default="artifacts/haraya_bert.pt", 
                        type=Path, help="Output path for trained model")
    
    args = parser.parse_args()
    
    # I-resolve ang paths mula sa script directory.
    script_dir = Path(__file__).resolve().parent
    combined_csv = args.combined_dataset
    if not combined_csv.is_absolute():
        combined_csv = script_dir / combined_csv
    
    output_path = args.output
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    
    training_csv = script_dir / "data" / "training_dataset.csv"
    
    # Ipakita ang training pipeline header.
    print("\n" + "="*60)
    print("HARAYA BERT Training Pipeline")
    print("="*60 + "\n")
    
    # Tingnan kung nandoon ang combined dataset.
    if not combined_csv.exists():
        print(f"❌ Error: {combined_csv} not found")
        return
    
    # Hakbang 1: Ihanda ang dataset para sa training.
    prepare_training_dataset(combined_csv, training_csv)
    
    # Hakbang 2: I-train ang BERT model gamit ang prepared data.
    print("\n🚀 Starting BERT training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Model: {CONFIG.model_name}")
    print(f"  Output: {output_path}\n")
    
    try:
        # I-execute ang training process mula sa train.py.
        train_model(training_csv, output_path, epochs=args.epochs, batch_size=args.batch_size)
        print(f"\n✅ Training complete! Model saved to {output_path}")
    except Exception as e:
        # Kung may error, ipakita ang message at i-raise.
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    # Simulan ang training pipeline mula dito kapag tinakbo ang script directly.
    main()
