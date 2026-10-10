
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

TARGET = "default_payment_next_month"
RANDOM_STATE = 42


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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

    print("Preparing the training data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # SHAP's tree explainer works most reliably with dense arrays here.
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_train_transformed = np.asarray(X_train_transformed)
    X_test_transformed = np.asarray(X_test_transformed)

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )

    print("Training Gradient Boosting...")
    model.fit(X_train_transformed, y_train)

    # Explain a reproducible sample of test records.
    sample_size = min(1000, len(X_test_transformed))
    X_explain = X_test_transformed[:sample_size]

    print(f"Calculating SHAP values for {sample_size} test records...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Handle SHAP versions that return a list for binary classifiers.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Handle newer APIs that may include a class-output dimension.
    if getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape: {shap_values.shape}"
        )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP output does not match the transformed feature count."
        )

    # Global importance: mean absolute SHAP value per transformed feature.
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap_value": np.abs(shap_values).mean(axis=0),
    }).sort_values(
        "mean_absolute_shap_value",
        ascending=False,
    )

    importance.to_csv(
        RESULTS_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTOP 15 SHAP FEATURES")
    print(importance.head(15).round(5).to_string(index=False))

    # Figure 1: global feature importance.
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Gradient Boosting: SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 2: direction and distribution of SHAP effects.
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("Gradient Boosting: SHAP Summary")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSHAP analysis completed.")
    print("Results: outputs/results/shap_feature_importance.csv")
    print("Figures: outputs/figures/shap_feature_importance.png")
    print("Figures: outputs/figures/shap_summary.png")


if __name__ == "__main__":
    main()