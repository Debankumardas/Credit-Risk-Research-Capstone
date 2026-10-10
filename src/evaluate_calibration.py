
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "default_payment_next_month"
RANDOM_STATE = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }

    results = []
    plt.figure(figsize=(8, 7))

    # Perfect calibration reference.
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for name, (classifier, scale_numeric) in models.items():
        numeric_transformer = (
            StandardScaler() if scale_numeric else "passthrough"
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
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

        print(f"Evaluating calibration: {name}...")
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        brier = brier_score_loss(y_test, probabilities)
        logloss = log_loss(y_test, probabilities)

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

        results.append({
            "model": name,
            "brier_score": brier,
            "log_loss": logloss,
            "test_rows": len(y_test),
            "calibration_bins": 10,
            "bin_strategy": "quantile",
        })

        print(f"  Brier score: {brier:.4f}")
        print(f"  Log loss:    {logloss:.4f}")

    plt.xlabel("Mean predicted default probability")
    plt.ylabel("Observed default frequency")
    plt.title("Probability Calibration on the Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "calibration_curves.png",
        dpi=300,
    )
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "calibration_metrics.csv",
        index=False,
    )

    print("\n--- CALIBRATION COMPARISON ---")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved: outputs/results/calibration_metrics.csv")
    print("Saved: outputs/figures/calibration_curves.png")


if __name__ == "__main__":
    main()