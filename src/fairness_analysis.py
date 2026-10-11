
from pathlib import Path

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
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 2026
THRESHOLD = 0.25


def subgroup_metrics(frame, group_column, predictions):
    rows = []

    evaluation = frame[[group_column, TARGET]].copy()
    evaluation["prediction"] = predictions

    for group, subset in evaluation.groupby(group_column, dropna=False):
        actual = subset[TARGET].astype(int)
        predicted = subset["prediction"].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        rows.append({
            "grouping": group_column,
            "group": str(group),
            "sample_size": int(len(subset)),
            "actual_default_rate": float(actual.mean()),
            "predicted_default_rate": float(predicted.mean()),
            "precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "recall": float(
                recall_score(actual, predicted, zero_division=0)
            ),
            "false_positive_rate": (
                float(fp / (fp + tn)) if (fp + tn) else None
            ),
            "false_negative_rate": (
                float(fn / (fn + tp)) if (fn + tp) else None
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    # Create interpretable age groups for descriptive analysis.
    age_bins = [0, 29, 39, 49, 59, float("inf")]
    age_labels = ["29 and under", "30-39", "40-49", "50-59", "60+"]

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )

    # Keep AGE_GROUP only for subgroup analysis, not model training.
    X = df.drop(columns=[TARGET, "AGE_GROUP"])
    y = df[TARGET].astype(int)

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # Preserve the same stratified 60/20/20 split.
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

    print("Training Gradient Boosting on the training split...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Match predictions to the exact rows in the final test set.
    test_frame = df.loc[X_test.index].copy()
    test_frame[TARGET] = y_test
    test_frame["AGE_GROUP"] = pd.cut(
        test_frame["AGE"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )

    sex_results = subgroup_metrics(test_frame, "SEX", predictions)
    age_results = subgroup_metrics(test_frame, "AGE_GROUP", predictions)

    sex_file = RESULTS_DIR / "fairness_by_sex.csv"
    age_file = RESULTS_DIR / "fairness_by_age_group.csv"

    sex_results.to_csv(sex_file, index=False)
    age_results.to_csv(age_file, index=False)

    print(f"\nSUBGROUP RESULTS BY SEX — threshold {THRESHOLD:.2f}")
    print(sex_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSUBGROUP RESULTS BY AGE GROUP — threshold {THRESHOLD:.2f}")
    print(age_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot recall by group to compare how often actual defaults are identified.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, title in [
        (axes[0], sex_results, "Recall by Sex"),
        (axes[1], age_results, "Recall by Age Group"),
    ]:
        ax.bar(results["group"], results["recall"])
        ax.set_title(title)
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(f"Subgroup Recall — Threshold {THRESHOLD:.2f}")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fairness_subgroup_recall.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:")
    print("outputs/results/fairness_by_sex.csv")
    print("outputs/results/fairness_by_age_group.csv")
    print("outputs/figures/fairness_subgroup_recall.png")
    print("\nNote: These are descriptive subgroup metrics, not proof of fairness.")


if __name__ == "__main__":
    main()