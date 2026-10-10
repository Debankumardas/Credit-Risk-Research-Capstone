
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Nominal categorical features.
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    # All remaining predictors, including repayment-status codes.
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # Stratification preserves the target proportions in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Training Logistic Regression baseline...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "Logistic Regression",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "threshold": 0.5,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "brier_score": float(
            brier_score_loss(y_test, probabilities)
        ),
    }

    with (RESULTS_DIR / "logistic_regression_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    predictions_df = pd.DataFrame({
        "actual_default": y_test.to_numpy(),
        "predicted_default": predictions,
        "predicted_probability": probabilities,
    })
    predictions_df.to_csv(
        RESULTS_DIR / "logistic_regression_test_predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No default", "Default"],
    )
    display.plot(values_format="d")
    plt.title("Logistic Regression: Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "logistic_regression_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    print("\n--- TEST-SET METRICS ---")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(matrix)
    print("\nMetrics saved to outputs/results/")
    print("Predictions saved to outputs/results/")
    print("Confusion matrix saved to outputs/figures/")


if __name__ == "__main__":
    main()