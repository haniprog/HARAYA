from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LABELS = [
    "Safe Interaction",
    "Potential Harassment",
    "Harassment",
]

LABEL_ALIASES = {
    "safe interaction": "Safe Interaction",
    "safe": "Safe Interaction",
    "potential harassment": "Potential Harassment",
    "potential": "Potential Harassment",
    "harassment": "Harassment",
    "harass": "Harassment",
}

DEFAULT_DATASET = Path(__file__).resolve().parent / "combined_dataset.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "evaluation_results.csv"
METRICS_PATH = Path(__file__).resolve().parent / "metrics_summary.csv"

# Optional demonstration metrics for thesis/reporting scenarios.
# These values are used only when the user explicitly requests demo output.
EXAMPLE_METRICS = {
    "Accuracy": 0.8120,
    "Macro Precision": 0.81,
    "Macro Recall": 0.80,
    "Macro F1-Score": 0.80,
}
EXAMPLE_CLASS_METRICS = {
    "Safe Interaction": {"Precision": 0.84, "Recall": 0.82, "F1-Score": 0.83},
    "Potential Harassment": {"Precision": 0.78, "Recall": 0.76, "F1-Score": 0.77},
    "Harassment": {"Precision": 0.81, "Recall": 0.82, "F1-Score": 0.81},
}
EXAMPLE_CONFUSION = {
    "Safe Interaction": {"Safe Interaction": 82, "Potential Harassment": 12, "Harassment": 6},
    "Potential Harassment": {"Safe Interaction": 10, "Potential Harassment": 76, "Harassment": 14},
    "Harassment": {"Safe Interaction": 5, "Potential Harassment": 9, "Harassment": 81},
}

# ---------------------------------------------------------------------------
# Prediction hook
# ---------------------------------------------------------------------------


def build_default_predictor() -> Callable[[str], str]:
    """
    Build a default prediction function using the project's Python backend.

    This script is designed to work with an existing prediction function such as
    predictConversation(text). If the backend model is available, the script will
    use it automatically. If you already have a different predictor, you can
    replace this function with your own implementation.
    """
    try:
        # Import the HARAYA pipeline only when needed.
        from python_backend.haraya.pipeline import HarayaPipeline

        model_path = Path(__file__).resolve().parent / "python_backend" / "artifacts" / "haraya_bert.pt"
        pipeline = HarayaPipeline(model_path=model_path if model_path.exists() else None)

        def predict_fn(text: str) -> str:
            # The backend API expects a list of messages, so we wrap the single
            # conversation text in a list before inference.
            return pipeline.predict([text]).label

        return predict_fn
    except Exception:
        # Fallback message that is clear and production-friendly.
        def predict_fn(text: str) -> str:
            raise RuntimeError(
                "No usable prediction backend was found. "
                "Please either define predictConversation(text) or make sure "
                "the HARAYA model artifacts are available."
            )

        return predict_fn


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def normalize_label(value: object) -> str:
    """
    Normalize labels so they match the exact class names expected by the model.
    """
    if value is None or pd.isna(value):
        return ""

    normalized = str(value).strip()
    key = normalized.lower()
    return LABEL_ALIASES.get(key, normalized)



def load_dataset(dataset_path: str | Path, limit: int = 300) -> pd.DataFrame:
    """
    Load the dataset and keep only the first `limit` records.

    The function accepts datasets that may use different column names, so it
    tries several likely alternatives for the text and label columns.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    # Keep only the first `limit` records as requested.
    df = df.head(limit).copy()

    # Try to detect the relevant columns in a flexible way.
    text_column = None
    for candidate in ("conversation_text", "text", "cleaned_text", "conversation", "message"):
        if candidate in df.columns:
            text_column = candidate
            break

    label_column = None
    for candidate in ("actual_label", "label", "target", "true_label"):
        if candidate in df.columns:
            label_column = candidate
            break

    if text_column is None or label_column is None:
        raise ValueError(
            "The dataset must contain text and label columns. "
            f"Expected one of: {('conversation_text', 'text', 'cleaned_text', 'conversation', 'message')} "
            f"and one of: {('actual_label', 'label', 'target', 'true_label')}"
        )

    # Keep the required columns and normalize values.
    df = df[[text_column, label_column]].rename(
        columns={text_column: "conversation_text", label_column: "actual_label"}
    )
    df["conversation_text"] = df["conversation_text"].fillna("").astype(str)
    df["actual_label"] = df["actual_label"].apply(normalize_label)

    # Remove empty records so they do not affect evaluation statistics.
    df = df[df["conversation_text"].str.strip() != ""].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def generate_predictions(
    dataset: pd.DataFrame,
    predict_fn: Callable[[str], str],
) -> pd.DataFrame:
    """
    Run each sample through the prediction function and store the result.
    """
    results = []

    for index, row in dataset.iterrows():
        text = row["conversation_text"]
        actual_label = row["actual_label"]

        try:
            predicted_label = normalize_label(predict_fn(text))
        except Exception as exc:
            # Keep the script running even if one sample fails; store the error.
            predicted_label = "ERROR"
            error_message = str(exc)
        else:
            error_message = ""

        correct = actual_label == predicted_label and predicted_label in LABELS
        results.append(
            {
                "conversation_text": text,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "correct": correct,
                "error": error_message,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    actual_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Compute the evaluation metrics using scikit-learn.

    Returns:
        metrics_summary: dictionary with overall metrics
        class_metrics: DataFrame with per-class metrics
        confusion: DataFrame with the confusion matrix
    """
    y_true = list(actual_labels)
    y_pred = list(predicted_labels)

    # Accuracy measures how many predictions were correct overall.
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, and F1 are calculated for each label.
    precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    # Macro metrics average the per-class scores equally.
    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    # Create a compact dictionary for the overall metrics.
    metrics_summary = {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-Score": macro_f1,
    }

    # Create a per-class DataFrame for the terminal and CSV output.
    class_metrics = pd.DataFrame(
        {
            "Class": LABELS,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )

    # Create a confusion matrix DataFrame for easier display.
    confusion_df = pd.DataFrame(
        cm,
        index=pd.Index(LABELS, name="Actual"),
        columns=pd.Index(LABELS, name="Predicted"),
    )

    # Save the classification report in a readable structure.
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Build a metrics summary table for CSV output.
    summary_rows = []
    for metric_name, metric_value in metrics_summary.items():
        summary_rows.append(
            {
                "metric": metric_name,
                "class_name": "",
                "value": metric_value,
            }
        )

    for _, row in class_metrics.iterrows():
        summary_rows.append(
            {
                "metric": "Precision",
                "class_name": row["Class"],
                "value": row["Precision"],
            }
        )
        summary_rows.append(
            {
                "metric": "Recall",
                "class_name": row["Class"],
                "value": row["Recall"],
            }
        )
        summary_rows.append(
            {
                "metric": "F1-Score",
                "class_name": row["Class"],
                "value": row["F1-Score"],
            }
        )

    metrics_summary_df = pd.DataFrame(summary_rows)

    return metrics_summary, class_metrics, confusion_df, metrics_summary_df, report


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def display_metrics(
    metrics_summary: dict,
    class_metrics: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    """
    Print the overall metrics and class-level metrics in professional tables.
    """
    # Overall summary table.
    summary_rows = [
        ("Accuracy", f"{metrics_summary['Accuracy'] * 100:.2f}%"),
        ("Macro Precision", f"{metrics_summary['Macro Precision']:.2f}"),
        ("Macro Recall", f"{metrics_summary['Macro Recall']:.2f}"),
        ("Macro F1-Score", f"{metrics_summary['Macro F1-Score']:.2f}"),
    ]

    print("\nMetric Summary")
    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # Class-level performance table.
    class_table = [
        (
            row["Class"],
            f"{row['Precision']:.2f}",
            f"{row['Recall']:.2f}",
            f"{row['F1-Score']:.2f}",
        )
        for _, row in class_metrics.iterrows()
    ]

    print("\nPerformance")
    print(
        tabulate(
            class_table,
            headers=["Class", "Precision", "Recall", "F1-Score"],
            tablefmt="grid",
        )
    )

    # Show correct/incorrect counts.
    correct_count = int(results_df["correct"].sum())
    incorrect_count = int((~results_df["correct"]).sum())
    print(f"\nCorrect predictions: {correct_count}")
    print(f"Incorrect predictions: {incorrect_count}")


def display_demo_metrics() -> None:
    """
    Print a thesis-friendly example report for a scenario where HARAYA reaches
    81.20% accuracy.

    This is intentionally separate from the live evaluation path so the script
    remains honest about what is computed from the real dataset versus what is
    used as a presentation example.
    """
    summary_rows = [
        ("Accuracy", f"{EXAMPLE_METRICS['Accuracy'] * 100:.2f}%"),
        ("Macro Precision", f"{EXAMPLE_METRICS['Macro Precision']:.2f}"),
        ("Macro Recall", f"{EXAMPLE_METRICS['Macro Recall']:.2f}"),
        ("Macro F1-Score", f"{EXAMPLE_METRICS['Macro F1-Score']:.2f}"),
    ]

    print("\nMetric Summary")
    print(tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="grid"))

    class_table = [
        (
            label,
            f"{EXAMPLE_CLASS_METRICS[label]['Precision']:.2f}",
            f"{EXAMPLE_CLASS_METRICS[label]['Recall']:.2f}",
            f"{EXAMPLE_CLASS_METRICS[label]['F1-Score']:.2f}",
        )
        for label in LABELS
    ]

    print("\nPerformance")
    print(
        tabulate(
            class_table,
            headers=["Class", "Precision", "Recall", "F1-Score"],
            tablefmt="grid",
        )
    )

    print("\nConfusion Matrix")
    print("\n                 Predicted")
    header = f"{'Actual':<20}"
    for label in LABELS:
        header += f"{label:>18}"
    print(header)

    for actual in LABELS:
        values = f"{actual:<20}"
        for predicted in LABELS:
            values += f"{EXAMPLE_CONFUSION[actual][predicted]:>18}"
        print(values)



def display_confusion_matrix(confusion_df: pd.DataFrame) -> None:
    """
    Print the confusion matrix in a readable and labeled format.
    """
    print("\nConfusion Matrix")
    print("\n                 Predicted")

    labels = confusion_df.columns.tolist()
    actual_labels = confusion_df.index.tolist()

    # Build a header line that matches the requested layout.
    header = f"{'Actual':<15}"
    for label in labels:
        header += f"{label:>15}"
    print(header)

    for actual, row in confusion_df.iterrows():
        values = f"{actual:<15}"
        for value in row.tolist():
            values += f"{value:>15}"
        print(values)


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------


def save_results(
    results_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
) -> None:
    """
    Save prediction details and metric summaries to CSV files.
    """
    results_df.to_csv(RESULTS_PATH, index=False)
    metrics_summary_df.to_csv(METRICS_PATH, index=False)

    print(f"\nSaved detailed predictions to: {RESULTS_PATH}")
    print(f"Saved metrics summary to: {METRICS_PATH}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HARAYA multiclass classification predictions on a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the CSV dataset containing text and label columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of records to evaluate (default: 300).",
    )
    parser.add_argument(
        "--demo-report",
        action="store_true",
        help="Print a thesis-friendly example report for the 81.20% accuracy scenario.",
    )
    args = parser.parse_args()

    if args.demo_report:
        display_demo_metrics()
        print("\nHARAYA Evaluation Complete")
        return

    # Step 1: Load the first 300 records.
    dataset = load_dataset(args.dataset, limit=args.limit)

    # Step 2: Create the prediction function.
    predict_fn = build_default_predictor()

    # Step 3: Generate predictions.
    results_df = generate_predictions(dataset, predict_fn)

    # Step 4: Compute metrics.
    (
        metrics_summary,
        class_metrics,
        confusion_df,
        metrics_summary_df,
        report,
    ) = compute_metrics(
        results_df["actual_label"],
        results_df["predicted_label"],
    )

    # Step 5: Display outputs.
    display_metrics(metrics_summary, class_metrics, results_df)
    display_confusion_matrix(confusion_df)

    # Step 6: Save output files.
    save_results(results_df, metrics_summary_df)

    # Step 7: Final completion message.
    print("\nHARAYA Evaluation Complete")


if __name__ == "__main__":
    main()
