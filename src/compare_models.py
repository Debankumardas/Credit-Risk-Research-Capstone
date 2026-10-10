
import json
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"

TARGET = "default_payment_next_month"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for model_name, classifier in models.items():
        print(f"\nTraining {model_name}...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "threshold": 0.5,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc_average_precision": average_precision_score(
                y_test, probabilities
            ),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "brier_score": brier_score_loss(
                y_test, probabilities
            ),
        }

        results.append(metrics)

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(
            "PR-AUC (Average Precision): "
            f"{metrics['pr_auc_average_precision']:.4f}"
        )
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"Brier score: {metrics['brier_score']:.4f}")

    # Include the already measured Logistic Regression baseline.
    baseline_file = RESULTS_DIR / "logistic_regression_metrics.json"
    with baseline_file.open("r", encoding="utf-8") as file:
        baseline = json.load(file)

    all_results = [
        {
            key: baseline[key]
            for key in [
                "model", "threshold", "roc_auc",
                "pr_auc_average_precision", "accuracy",
                "precision", "recall", "f1_score", "brier_score",
            ]
        },
        *results,
    ]

    comparison = pd.DataFrame(all_results)
    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n--- MODEL COMPARISON ---")
    print(
        comparison[
            [
                "model", "roc_auc",
                "pr_auc_average_precision",
                "recall", "f1_score", "brier_score",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved: outputs/results/model_comparison.csv")


if __name__ == "__main__":
    main()