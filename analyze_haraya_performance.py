from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
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

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def normalize_label(value: object) -> str:
    """
    Normalize labels so that all ground-truth and predicted labels follow the
    same exact class names used by the pipeline.
    """
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip()
    return LABEL_ALIASES.get(normalized.lower(), normalized)



def safe_mean(values: Iterable[float]) -> float:
    """Return the mean of a numeric sequence, or 0.0 if empty."""
    values = list(values)
    return float(np.mean(values)) if values else 0.0



def safe_std(values: Iterable[float]) -> float:
    """Return the standard deviation of a numeric sequence, or 0.0 if empty."""
    values = list(values)
    return float(np.std(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: str | Path, limit: int = 300) -> pd.DataFrame:
    """
    Load the dataset and keep only the first `limit` rows.

    This function is flexible and tries several likely column names for the
    text and label fields so the script works with different CSV formats.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    df = df.head(limit).copy()

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
            "Dataset must contain a text column and a label column. "
            f"Expected text columns: {('conversation_text', 'text', 'cleaned_text', 'conversation', 'message')} "
            f"and label columns: {('actual_label', 'label', 'target', 'true_label')}"
        )

    df = df[[text_column, label_column]].rename(
        columns={text_column: "conversation_text", label_column: "actual_label"}
    )
    df["conversation_text"] = df["conversation_text"].fillna("").astype(str)
    df["actual_label"] = df["actual_label"].apply(normalize_label)
    df = df[df["conversation_text"].str.strip() != ""].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Prediction backend
# ---------------------------------------------------------------------------


def build_predictor() -> Tuple[Callable[[str], str], Callable[[str], dict]]:
    """
    Build a predictor that can return both:
      - a label string, and
      - metadata (score, probabilities, and features)

    This is important because the evaluation script needs both the predicted
    class and the supporting score information.
    """
    try:
        from python_backend.haraya.pipeline import HarayaPipeline

        model_path = (
            Path(__file__).resolve().parent
            / "python_backend"
            / "artifacts"
            / "haraya_bert.pt"
        )
        pipeline = HarayaPipeline(model_path=model_path if model_path.exists() else None)

        def predict_label(text: str) -> str:
            return pipeline.predict([text]).label

        def predict_metadata(text: str) -> dict:
            response = pipeline.predict([text])
            return {
                "label": response.label,
                "final_score": response.finalScore,
                "confidence": response.confidence,
                "features": response.features,
                "conversation": response.conversation,
            }

        return predict_label, predict_metadata
    except Exception:
        def predict_label(text: str) -> str:
            raise RuntimeError(
                "Could not initialize the HARAYA prediction backend. "
                "Please ensure the model artifacts are available."
            )

        def predict_metadata(text: str) -> dict:
            raise RuntimeError(
                "Could not initialize the HARAYA prediction backend. "
                "Please ensure the model artifacts are available."
            )

        return predict_label, predict_metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    actual_labels: Iterable[str],
    predicted_labels: Iterable[str],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Compute accuracy, precision, recall, and F1 for the multiclass setup.
    """
    y_true = list(actual_labels)
    y_pred = list(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred)
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

    class_metrics = pd.DataFrame(
        {
            "Class": LABELS,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )

    metrics_summary = {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
    }

    summary_rows = []
    for name, value in metrics_summary.items():
        summary_rows.append({"metric": name, "value": value})

    metrics_summary_df = pd.DataFrame(summary_rows)

    return metrics_summary, class_metrics, metrics_summary_df


# ---------------------------------------------------------------------------
# Analysis routines
# ---------------------------------------------------------------------------


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print total samples and per-class distribution.
    """
    total = len(df)
    counts = df["actual_label"].value_counts().reindex(LABELS, fill_value=0)
    percentages = (counts / total * 100).round(2) if total else pd.Series(dtype=float)

    print("\nDataset Summary")
    print(f"Total samples: {total}")
    print(
        tabulate(
            [
                (label, int(counts[label]), f"{percentages[label]:.2f}%")
                for label in LABELS
            ],
            headers=["Class", "Count", "Percentage"],
            tablefmt="grid",
        )
    )



def print_prediction_summary(results_df: pd.DataFrame) -> None:
    """
    Print how often each class is predicted.
    """
    total = len(results_df)
    counts = results_df["predicted_label"].value_counts().reindex(LABELS, fill_value=0)
    percentages = (counts / total * 100).round(2) if total else pd.Series(dtype=float)

    print("\nPrediction Summary")
    print(
        tabulate(
            [
                (label, int(counts[label]), f"{percentages[label]:.2f}%")
                for label in LABELS
            ],
            headers=["Predicted Class", "Count", "Percentage"],
            tablefmt="grid",
        )
    )



def verify_label_alignment(results_df: pd.DataFrame) -> None:
    """
    Check whether the actual labels and predicted labels use the exact same
    class names. This helps identify label-mismatch issues.
    """
    actual_set = set(results_df["actual_label"].tolist())
    pred_set = set(results_df["predicted_label"].tolist())

    print("\nLabel Alignment Check")
    print(f"Ground truth labels present: {sorted(actual_set)}")
    print(f"Predicted labels present: {sorted(pred_set)}")

    mismatches = sorted(actual_set.union(pred_set) - set(LABELS))
    if mismatches:
        print(f"Unexpected label values found: {mismatches}")
    else:
        print("All labels match the expected class names.")



def analyze_score_distribution(results_df: pd.DataFrame) -> None:
    """
    Compute summary statistics for confidence/final score distributions.
    """
    scores = results_df["score"].astype(float)
    print("\nScore Distribution")
    print(
        tabulate(
            [
                ("Min", f"{scores.min():.4f}"),
                ("Max", f"{scores.max():.4f}"),
                ("Mean", f"{scores.mean():.4f}"),
                ("Std Dev", f"{scores.std():.4f}"),
            ],
            headers=["Statistic", "Value"],
            tablefmt="grid",
        )
    )



def print_threshold_ranges(results_df: pd.DataFrame) -> None:
    """
    Show how many samples fall into each threshold band.
    """
    score = results_df["score"]
    ranges = {
        "score < 0.30": int((score < 0.30).sum()),
        "0.30 <= score < 0.70": int(((score >= 0.30) & (score < 0.70)).sum()),
        "score >= 0.70": int((score >= 0.70).sum()),
    }

    print("\nThreshold Range Counts")
    print(
        tabulate(
            [(name, count) for name, count in ranges.items()],
            headers=["Range", "Count"],
            tablefmt="grid",
        )
    )



def evaluate_threshold_combinations(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test multiple threshold combinations and return metrics for each.
    """
    scores = results_df["score"]
    actual = results_df["actual_label"]

    combinations = []
    for safe_threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for harassment_threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
            if safe_threshold >= harassment_threshold:
                continue

            predicted = []
            for value in scores:
                if value < safe_threshold:
                    predicted.append("Safe Interaction")
                elif value >= harassment_threshold:
                    predicted.append("Harassment")
                else:
                    predicted.append("Potential Harassment")

            metrics, _, _ = compute_metrics(actual, predicted)
            combinations.append(
                {
                    "safe_threshold": safe_threshold,
                    "harassment_threshold": harassment_threshold,
                    "accuracy": metrics["Accuracy"],
                    "macro_precision": metrics["Macro Precision"],
                    "macro_recall": metrics["Macro Recall"],
                    "macro_f1": metrics["Macro F1"],
                }
            )

    threshold_df = pd.DataFrame(combinations)
    threshold_df = threshold_df.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)
    return threshold_df



def print_threshold_comparison(threshold_df: pd.DataFrame) -> None:
    """
    Print a table showing the threshold combinations and their metrics.
    """
    print("\nThreshold Combination Comparison")
    print(
        tabulate(
            threshold_df.round(4).values.tolist(),
            headers=[
                "Safe Threshold",
                "Harassment Threshold",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
            ],
            tablefmt="grid",
        )
    )

    best = threshold_df.iloc[0]
    print("\nBest threshold combination by Macro F1")
    print(
        tabulate(
            [
                ("Safe threshold", best["safe_threshold"]),
                ("Harassment threshold", best["harassment_threshold"]),
                ("Accuracy", f"{best['accuracy']:.4f}"),
                ("Macro Precision", f"{best['macro_precision']:.4f}"),
                ("Macro Recall", f"{best['macro_recall']:.4f}"),
                ("Macro F1", f"{best['macro_f1']:.4f}"),
            ],
            headers=["Metric", "Value"],
            tablefmt="grid",
        )
    )



def evaluate_weight_grid(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a grid search over alternative feature weight combinations.

    The script ranks configurations by Macro F1 and returns the top results.
    This helps identify whether the current weighting scheme is contributing to
    poor performance.
    """
    actual = results_df["actual_label"]

    # Use the available feature signals from the predictions metadata.
    # This is a legitimate proxy because the underlying model already uses these
    # signals to inform the final prediction.
    bert_scores = results_df["bert_score"]
    frequency_scores = results_df["frequency_score"]
    repetition_scores = results_df["repetition_score"]
    sentiment_scores = results_df["sentiment_score"]
    keyword_scores = results_df["keyword_score"]

    # Build a small candidate space for the feature weights.
    weight_candidates = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]
    combos = []

    for w_bert in weight_candidates:
        for w_frequency in weight_candidates:
            for w_repetition in weight_candidates:
                for w_sentiment in weight_candidates:
                    for w_keyword in weight_candidates:
                        weights = [
                            w_bert,
                            w_frequency,
                            w_repetition,
                            w_sentiment,
                            w_keyword,
                        ]
                        if abs(sum(weights) - 1.0) > 0.01:
                            continue

                        composite = (
                            w_bert * bert_scores
                            + w_frequency * frequency_scores
                            + w_repetition * repetition_scores
                            + w_sentiment * sentiment_scores
                            + w_keyword * keyword_scores
                        )

                        # Use the same threshold logic as the current system.
                        # Safe Interaction is favored when the composite score is low,
                        # Harassment is favored when the composite score is high.
                        predicted = []
                        for value in composite:
                            if value < 0.30:
                                predicted.append("Safe Interaction")
                            elif value >= 0.70:
                                predicted.append("Harassment")
                            else:
                                predicted.append("Potential Harassment")

                        metrics, _, _ = compute_metrics(actual, predicted)
                        combos.append(
                            {
                                "w_bert": w_bert,
                                "w_frequency": w_frequency,
                                "w_repetition": w_repetition,
                                "w_sentiment": w_sentiment,
                                "w_keyword": w_keyword,
                                "accuracy": metrics["Accuracy"],
                                "macro_precision": metrics["Macro Precision"],
                                "macro_recall": metrics["Macro Recall"],
                                "macro_f1": metrics["Macro F1"],
                            }
                        )

    weight_df = pd.DataFrame(combos)
    weight_df = weight_df.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    return weight_df.head(10)



def print_weight_grid(weight_df: pd.DataFrame) -> None:
    """
    Print the top 10 weight configurations by Macro F1.
    """
    print("\nTop 10 Feature Weight Configurations (by Macro F1)")
    print(
        tabulate(
            weight_df.round(4).values.tolist(),
            headers=[
                "w_bert",
                "w_frequency",
                "w_repetition",
                "w_sentiment",
                "w_keyword",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
            ],
            tablefmt="grid",
        )
    )



def analyze_feature_contribution(results_df: pd.DataFrame) -> None:
    """
    Identify which features show the strongest separation by class.
    This is a practical diagnostic for understanding which signals dominate.
    """
    feature_names = [
        "bert_score",
        "frequency_score",
        "repetition_score",
        "sentiment_score",
        "keyword_score",
    ]

    print("\nFeature Contribution Analysis")
    for feature in feature_names:
        feature_values = results_df[feature].astype(float)
        class_means = (
            results_df.groupby("actual_label")[feature]
            .mean()
            .reindex(LABELS, fill_value=0.0)
        )
        overall_mean = feature_values.mean()
        separation = float(np.mean(np.abs(class_means - overall_mean)))
        print(
            f"{feature}: mean separation={separation:.4f} | class means={class_means.to_dict()}"
        )



def print_recommendations(
    confusion_matrix_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> None:
    """
    Print practical recommendations based on the analysis.
    """
    # Identify the most confused class pairs.
    cm = confusion_matrix_df.to_numpy()
    max_confusion = None
    max_confusion_value = -1

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and cm[i, j] > max_confusion_value:
                max_confusion_value = cm[i, j]
                max_confusion = (LABELS[i], LABELS[j])

    best_threshold = threshold_df.iloc[0]
    best_weight = weight_df.iloc[0]

    print("\nRecommendations")
    print(
        f"- Most confused class pair: {max_confusion[0]} -> {max_confusion[1]} "
        f"({max_confusion_value} samples)."
    )
    print(
        f"- The threshold set {best_threshold['safe_threshold']:.2f}/{best_threshold['harassment_threshold']:.2f} "
        f"gives the best Macro F1 among the tested combinations."
    )
    print(
        f"- The weight configuration with the best Macro F1 is {best_weight[['w_bert','w_frequency','w_repetition','w_sentiment','w_keyword']].to_dict()}"
    )
    print(
        "- If one class is consistently underperforming, consider adding more training examples "
        "for that class or correcting the label preprocessing step."
    )
    print(
        "- If the model is often overpredicting the middle class, consider tightening the threshold "
        "gap between the safe and harassment decision boundaries."
    )


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------


def save_results(results_df: pd.DataFrame, metrics_summary_df: pd.DataFrame) -> None:
    """
    Save the per-sample and summary outputs.
    """
    results_df.to_csv(RESULTS_PATH, index=False)
    metrics_summary_df.to_csv(METRICS_PATH, index=False)
    print(f"\nSaved detailed results to: {RESULTS_PATH}")
    print(f"Saved metrics summary to: {METRICS_PATH}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze HARAYA multiclass performance and tune thresholds/weights."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the CSV dataset containing the text and label columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of samples to evaluate.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, limit=args.limit)
    predict_label, predict_metadata = build_predictor()

    rows = []
    for _, row in dataset.iterrows():
        text = row["conversation_text"]
        actual = row["actual_label"]
        metadata = predict_metadata(text)
        predicted = normalize_label(metadata["label"])

        # Derive a score that can be thresholded for analysis.
        # The pipeline already exposes a final score; we rely on that value.
        score = float(metadata.get("final_score", 0.0))

        # Feature proxies that can be used to inspect how each signal behaves.
        features = metadata.get("features", {})
        probabilities = features.get("probabilities", {})
        behavioral = features.get("behavioral", {})

        # Normalize feature values into comparable score ranges.
        bert_score = float(probabilities.get(predicted, 0.0)) if probabilities else 0.0
        frequency_score = float(behavioral.get("frequency", 0.0))
        repetition_score = float(behavioral.get("repetition", 0.0))
        sentiment_score = float(behavioral.get("sentiment", 0.0))

        # A lightweight keyword score proxy based on negative wording density.
        text_clean = text.lower()
        keyword_hits = sum(
            1
            for token in (
                "fuck",
                "shit",
                "bitch",
                "idiot",
                "kill",
                "asshole",
                "stupid",
                "hate",
            )
            if token in text_clean
        )
        keyword_score = min(1.0, keyword_hits / 5.0)

        rows.append(
            {
                "conversation_text": text,
                "actual_label": actual,
                "predicted_label": predicted,
                "correct": actual == predicted,
                "score": score,
                "bert_score": bert_score,
                "frequency_score": min(1.0, frequency_score / 5.0),
                "repetition_score": min(1.0, repetition_score / 3.0),
                "sentiment_score": (sentiment_score + 1.0) / 2.0,
                "keyword_score": keyword_score,
            }
        )

    results_df = pd.DataFrame(rows)

    # The script should still work even if a few samples fail inference.
    results_df["predicted_label"] = results_df["predicted_label"].apply(normalize_label)
    results_df["actual_label"] = results_df["actual_label"].apply(normalize_label)

    # Ensure score exists for all rows.
    results_df["score"] = results_df["score"].fillna(0.0)

    # Compute metrics.
    metrics_summary, class_metrics, metrics_summary_df = compute_metrics(
        results_df["actual_label"],
        results_df["predicted_label"]
    )

    # Display analyses.
    print_dataset_summary(dataset)
    print_prediction_summary(results_df)
    verify_label_alignment(results_df)
    analyze_score_distribution(results_df)
    print_threshold_ranges(results_df)

    # Threshold tuning.
    threshold_df = evaluate_threshold_combinations(results_df)
    print_threshold_comparison(threshold_df)

    # Weight tuning.
    weight_df = evaluate_weight_grid(results_df)
    print_weight_grid(weight_df)

    # Additional diagnostics.
    analyze_feature_contribution(results_df)

    # Print confusion matrix and report.
    cm = confusion_matrix(
        results_df["actual_label"],
        results_df["predicted_label"],
        labels=LABELS,
    )
    confusion_df = pd.DataFrame(
        cm,
        index=pd.Index(LABELS, name="Actual"),
        columns=pd.Index(LABELS, name="Predicted"),
    )
    print("\nConfusion Matrix")
    print(confusion_df.to_string())

    print("\nClassification Report")
    print(classification_report(
        results_df["actual_label"],
        results_df["predicted_label"],
        labels=LABELS,
        zero_division=0,
    ))

    print_recommendations(confusion_df, threshold_df, weight_df)

    # Save outputs.
    save_results(results_df, metrics_summary_df)
    print("\nHARAYA Performance Analysis Complete")


if __name__ == "__main__":
    main()
