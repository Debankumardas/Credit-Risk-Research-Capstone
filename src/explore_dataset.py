
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "credit_card_default.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

TARGET = "default_payment_next_month"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing values: {df.isna().sum().sum()}")

    print("\nTARGET DISTRIBUTION")
    print(df[TARGET].value_counts().sort_index())
    print("\nTarget percentages:")
    print((df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))

    # 1. Descriptive statistics
    df.describe().T.to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    # 2. Default vs non-default counts
    target_counts = (
        df[TARGET]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=["No default", "Default"],
        y=target_counts.values,
    )
    ax.bar_label(ax.containers[0], fmt="%.0f")
    plt.title("Credit Card Default: Target Distribution")
    plt.xlabel("Outcome")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # 3. Age distribution by outcome
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="AGE",
        hue=TARGET,
        bins=25,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Age Distribution by Default Outcome")
    plt.xlabel("Age")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_by_default.png", dpi=300)
    plt.close()

    # 4. Credit limit by outcome
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x=TARGET,
        y="LIMIT_BAL",
    )
    plt.title("Credit Limit by Default Outcome")
    plt.xlabel("Default next month (0 = No, 1 = Yes)")
    plt.ylabel("Credit limit")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "credit_limit_by_default.png", dpi=300)
    plt.close()

    # 5. Default rate by most recent repayment status
    # PAY_0 represents the repayment status for the most recent month.
    default_rates = (
        df.groupby("PAY_0", observed=True)[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("PAY_0")
    )
    default_rates["default_rate_percent"] = default_rates["mean"] * 100
    default_rates.to_csv(
        RESULTS_DIR / "default_rate_by_pay0.csv",
        index=False,
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=default_rates,
        x="PAY_0",
        y="default_rate_percent",
    )
    plt.title("Observed Default Rate by Recent Repayment Status")
    plt.xlabel("PAY_0 repayment status code")
    plt.ylabel("Observed default rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_by_pay0.png", dpi=300)
    plt.close()

    print("\nDEFAULT RATE BY PAY_0")
    print(
        default_rates[
            ["PAY_0", "count", "default_rate_percent"]
        ].round(2).to_string(index=False)
    )

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()