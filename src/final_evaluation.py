
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 2026


def build_preprocessor(numeric_features, categorical_features, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def build_models(numeric_features, categorical_features):
    logistic_preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=True
    )
    tree_preprocessor = build_preprocessor(
        numeric_features, categorical_features, scale_numeric=False
    )

    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", logistic_preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_model(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "accuracy_at_0_5": accuracy_score(y_test, predictions),
        "precision_at_0_5": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_at_0_5": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_at_0_5": f1_score(y_test, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
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

    # Reserve 20% as the final test set.
    # Split the remaining 80% into 60% training and 20% validation
    # of the complete dataset.
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

    split_summary = {
        "random_state": RANDOM_STATE,
        "total_rows": len(df),
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "final_test_rows": len(X_test),
        "training_default_rate": float(y_train.mean()),
        "validation_default_rate": float(y_validation.mean()),
        "final_test_default_rate": float(y_test.mean()),
        "note": (
            "This is a new holdout split, but earlier exploratory work "
            "used other splits of the same dataset. It is not a completely "
            "unseen external dataset."
        ),
    }

    print("Dataset split:")
    print(json.dumps(split_summary, indent=2))

    with open(RESULTS_DIR / "final_split_summary.json", "w") as file:
        json.dump(split_summary, file, indent=2)

    models = build_models(numeric_features, categorical_features)
    rows = []

    # Validation results support comparison before final test evaluation.
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        validation_probabilities = model.predict_proba(X_validation)[:, 1]
        validation_predictions = (validation_probabilities >= 0.5).astype(int)

        validation_row = {
            "model": name,
            "validation_roc_auc": roc_auc_score(
                y_validation, validation_probabilities
            ),
            "validation_average_precision": average_precision_score(
                y_validation, validation_probabilities
            ),
            "validation_brier_score": brier_score_loss(
                y_validation, validation_probabilities
            ),
            "validation_recall_at_0_5": recall_score(
                y_validation, validation_predictions, zero_division=0
            ),
        }

        rows.append(validation_row)

    validation_results = pd.DataFrame(rows)
    validation_results.to_csv(
        RESULTS_DIR / "final_validation_metrics.csv", index=False
    )

    print("\nValidation comparison:")
    print(validation_results.round(4).to_string(index=False))

    # Freeze the selection rule before opening the final test results:
    # select the model with the highest validation ROC-AUC.
    selected_name = validation_results.sort_values(
        "validation_roc_auc", ascending=False
    ).iloc[0]["model"]

    print(f"\nSelected using validation ROC-AUC: {selected_name}")

    # Refit each model on training + validation data, then evaluate
    # the frozen model selection on the final holdout.
    X_development_full = pd.concat([X_train, X_validation])
    y_development_full = pd.concat([y_train, y_validation])

    final_rows = []
    for name, model in models.items():
        model.fit(X_development_full, y_development_full)
        result = evaluate_model(name, model, X_test, y_test)
        result["selected_by_validation"] = name == selected_name
        final_rows.append(result)

    final_results = pd.DataFrame(final_rows)
    final_results.drop(columns=["confusion_matrix"]).to_csv(
        RESULTS_DIR / "final_test_metrics.csv", index=False
    )

    with open(RESULTS_DIR / "final_test_confusion_matrices.json", "w") as file:
        json.dump(
            {
                row["model"]: row["confusion_matrix"]
                for row in final_rows
            },
            file,
            indent=2,
        )

    print("\nFinal test results:")
    print(final_results.drop(columns=["confusion_matrix"]).round(4).to_string(
        index=False
    ))

    # Calibration plot for the final test set.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        fraction_positive, mean_predicted = calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=name,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Final Holdout Calibration Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "final_test_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nFinal evaluation completed.")
    print("Results saved in outputs/results/")
    print("Calibration figure saved in outputs/figures/final_test_calibration.png")


if __name__ == "__main__":
    main()