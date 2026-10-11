
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 2026


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

    # Reproduce the same stratified 60% / 20% / 20% split
    # used in final_evaluation.py.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
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

    print("Training Gradient Boosting on the training split...")
    model.fit(X_train, y_train)

    print("Generating predictions for the validation split...")
    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = [
        round(value, 2)
        for value in [0.10, 0.15, 0.20, 0.25, 0.30,
                      0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    ]

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_validation, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_validation, predictions, zero_division=0
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "predicted_default_count": int(tp + fp),
        })

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR / "threshold_analysis_validation.csv",
        index=False,
    )

    print("\nValidation threshold comparison:")
    print(results.round(4).to_string(index=False))

    # Illustrate precision and recall trade-offs.
    plt.figure(figsize=(9, 6))
    plt.plot(
        results["threshold"],
        results["precision"],
        marker="o",
        label="Precision",
    )
    plt.plot(
        results["threshold"],
        results["recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        results["threshold"],
        results["f1"],
        marker="o",
        label="F1-score",
    )

    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Gradient Boosting Threshold Analysis — Validation Set")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "threshold_analysis_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # This is a descriptive choice, not a universal lending policy.
    best_f1_row = results.loc[results["f1"].idxmax()]

    print("\nThreshold with the highest validation F1-score:")
    print(best_f1_row.round(4).to_string())

    print("\nSaved:")
    print("outputs/results/threshold_analysis_validation.csv")
    print("outputs/figures/threshold_analysis_validation.png")
    print("\nThe final test set was not used for threshold selection.")


if __name__ == "__main__":
    main()