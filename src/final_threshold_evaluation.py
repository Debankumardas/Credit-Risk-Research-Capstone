
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 2026
SELECTED_THRESHOLD = 0.25
REFERENCE_THRESHOLD = 0.50


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # Reproduce the 60% training, 20% validation, 20% test split.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        )),
    ])

    # Train once. Use the same fitted model for both thresholds.
    print("Training Gradient Boosting on the training split...")
    model.fit(X_train, y_train)

    # Select the operating threshold using validation data.
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    validation_predictions = (
        validation_probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    validation_f1 = f1_score(
        y_validation, validation_predictions, zero_division=0
    )

    print(f"Selected validation threshold: {SELECTED_THRESHOLD:.2f}")
    print(f"Validation F1 at threshold 0.25: {validation_f1:.4f}")

    # Evaluate both thresholds on the same final test records.
    print("\nEvaluating both thresholds on the final test set...")
    test_probabilities = model.predict_proba(X_test)[:, 1]

    comparison_rows = []
    for threshold in [SELECTED_THRESHOLD, REFERENCE_THRESHOLD]:
        row = calculate_metrics(y_test, test_probabilities, threshold)
        row["model"] = "Gradient Boosting"
        row["final_test_rows"] = int(len(y_test))
        row["final_test_default_rate"] = float(y_test.mean())
        row["roc_auc"] = float(
            roc_auc_score(y_test, test_probabilities)
        )
        row["average_precision"] = float(
            average_precision_score(y_test, test_probabilities)
        )
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    # Save the side-by-side comparison.
    comparison_file = RESULTS_DIR / "final_threshold_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\nFINAL TEST THRESHOLD COMPARISON")
    print(
        comparison[
            [
                "threshold", "accuracy", "precision", "recall", "f1",
                "true_negatives", "false_positives",
                "false_negatives", "true_positives",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    # Preserve the detailed JSON for the validation-selected threshold.
    selected = comparison_rows[0].copy()
    selected["threshold_selected_on_validation"] = SELECTED_THRESHOLD
    selected["validation_f1_at_selected_threshold"] = float(validation_f1)

    with open(
        RESULTS_DIR / "final_threshold_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selected, file, indent=2)

    # Classification report for the validation-selected threshold.
    selected_predictions = (
        test_probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    print("\nCLASSIFICATION REPORT — THRESHOLD 0.25")
    print(classification_report(
        y_test,
        selected_predictions,
        labels=[0, 1],
        target_names=["No default", "Default"],
        zero_division=0,
    ))

    # Plot confusion matrices for both thresholds side by side.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, threshold in zip(
        axes, [SELECTED_THRESHOLD, REFERENCE_THRESHOLD]
    ):
        predictions = (test_probabilities >= threshold).astype(int)
        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(
            [0, 1],
            labels=["Predicted no default", "Predicted default"],
        )
        ax.set_yticks(
            [0, 1],
            labels=["Actual no default", "Actual default"],
        )
        ax.set_title(f"Threshold {threshold:.2f}")

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, f"{matrix[row, col]:,}",
                    ha="center", va="center", color="black",
                )

        fig.colorbar(image, ax=ax)

    fig.suptitle("Final Test Confusion Matrices by Threshold")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "final_threshold_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:")
    print("outputs/results/final_threshold_comparison.csv")
    print("outputs/results/final_threshold_metrics.json")
    print("outputs/figures/final_threshold_comparison.png")


if __name__ == "__main__":
    main()