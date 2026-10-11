
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
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 2026
THRESHOLD = 0.25


def add_wilson_interval(successes, total):
    """Return a rate and its 95% Wilson confidence interval."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")

    rate = successes / total
    lower, upper = proportion_confint(
        count=successes,
        nobs=total,
        alpha=0.05,
        method="wilson",
    )
    return float(rate), float(lower), float(upper)


def analyze_group(frame, group_column):
    rows = []

    for group, subset in frame.groupby(group_column, observed=True):
        actual = subset[TARGET].astype(int)
        predicted = subset["prediction"].astype(int)

        # Recall: true positives / all actual defaults.
        actual_defaults = int((actual == 1).sum())
        true_positives = int(((actual == 1) & (predicted == 1)).sum())
        recall, recall_low, recall_high = add_wilson_interval(
            true_positives, actual_defaults
        )

        # False-positive rate: false positives / all actual non-defaults.
        actual_nondefaults = int((actual == 0).sum())
        false_positives = int(((actual == 0) & (predicted == 1)).sum())
        fpr, fpr_low, fpr_high = add_wilson_interval(
            false_positives, actual_nondefaults
        )

        rows.append({
            "grouping": group_column,
            "group": str(group),
            "sample_size": int(len(subset)),
            "actual_defaults": actual_defaults,
            "actual_nondefaults": actual_nondefaults,
            "recall": recall,
            "recall_ci_lower": recall_low,
            "recall_ci_upper": recall_high,
            "false_positive_rate": fpr,
            "fpr_ci_lower": fpr_low,
            "fpr_ci_upper": fpr_high,
        })

    return pd.DataFrame(rows)


def plot_intervals(results, metric, lower_col, upper_col, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    x_positions = range(len(results))
    estimates = results[metric].to_numpy()
    lower_errors = estimates - results[lower_col].to_numpy()
    upper_errors = results[upper_col].to_numpy() - estimates

    ax.errorbar(
        list(x_positions),
        estimates,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=5,
    )

    ax.set_xticks(list(x_positions), results["group"], rotation=25, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    age_bins = [0, 29, 39, 49, 59, float("inf")]
    age_labels = ["29 and under", "30-39", "40-49", "50-59", "60+"]

    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )

    X = df.drop(columns=[TARGET, "AGE_GROUP"])
    y = df[TARGET].astype(int)

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # Reproduce the established stratified 60/20/20 split.
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

    # Align the test labels and predictions with their original rows.
    test_frame = df.loc[X_test.index].copy()
    test_frame[TARGET] = y_test
    test_frame["prediction"] = predictions

    sex_results = analyze_group(test_frame, "SEX")
    age_results = analyze_group(test_frame, "AGE_GROUP")

    all_results = pd.concat(
        [sex_results, age_results],
        ignore_index=True,
    )

    output_csv = RESULTS_DIR / "fairness_uncertainty_intervals.csv"
    all_results.to_csv(output_csv, index=False)

    print("\nFAIRNESS METRICS WITH 95% WILSON CONFIDENCE INTERVALS")
    print(
        all_results[
            [
                "grouping", "group", "sample_size",
                "recall", "recall_ci_lower", "recall_ci_upper",
                "false_positive_rate", "fpr_ci_lower", "fpr_ci_upper",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    plot_intervals(
        sex_results,
        "recall",
        "recall_ci_lower",
        "recall_ci_upper",
        "Recall by Sex with 95% Wilson Intervals",
        FIGURES_DIR / "fairness_recall_ci_by_sex.png",
    )

    plot_intervals(
        age_results,
        "recall",
        "recall_ci_lower",
        "recall_ci_upper",
        "Recall by Age Group with 95% Wilson Intervals",
        FIGURES_DIR / "fairness_recall_ci_by_age.png",
    )

    plot_intervals(
        sex_results,
        "false_positive_rate",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "False-Positive Rate by Sex with 95% Wilson Intervals",
        FIGURES_DIR / "fairness_fpr_ci_by_sex.png",
    )

    plot_intervals(
        age_results,
        "false_positive_rate",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "False-Positive Rate by Age Group with 95% Wilson Intervals",
        FIGURES_DIR / "fairness_fpr_ci_by_age.png",
    )

    print("\nSaved:")
    print("outputs/results/fairness_uncertainty_intervals.csv")
    print("outputs/figures/fairness_recall_ci_by_sex.png")
    print("outputs/figures/fairness_recall_ci_by_age.png")
    print("outputs/figures/fairness_fpr_ci_by_sex.png")
    print("outputs/figures/fairness_fpr_ci_by_age.png")
    print("\nInterpretation note: intervals quantify sampling uncertainty")
    print("under a binomial approximation; they do not prove fairness.")


if __name__ == "__main__":
    main()