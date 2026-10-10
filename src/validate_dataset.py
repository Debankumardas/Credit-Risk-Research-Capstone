
"""
Validate the local UCI Default of Credit Card Clients dataset.

Run from the project root:
    python src/validate_dataset.py
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "credit_card_default.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "credit_card_default.csv"
RESULTS_DIR = ROOT / "outputs" / "results"
REPORT_FILE = RESULTS_DIR / "dataset_validation_report.json"

EXPECTED_ROWS = 30_000
TARGET_COLUMN = "default_payment_next_month"

FEATURE_NAMES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Place credit_card_default.csv inside data/raw/."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading the existing local dataset...")
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Rename generic feature columns when the CSV uses X1 through X23.
    generic_names = [f"X{i}" for i in range(1, 24)]

    if all(name in df.columns for name in generic_names):
        df = df.rename(
            columns=dict(zip(generic_names, FEATURE_NAMES))
        )

    expected_columns = FEATURE_NAMES + [TARGET_COLUMN]
    schema_valid = (
        len(df.columns) == len(expected_columns)
        and set(df.columns) == set(expected_columns)
    )

    if schema_valid:
        df = df[expected_columns]

    target = df[TARGET_COLUMN]
    target_numeric = pd.to_numeric(target, errors="coerce")
    target_values = sorted(target_numeric.dropna().unique().tolist())

    missing_by_column = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    checks = {
        "expected_row_count": bool(len(df) == EXPECTED_ROWS),
        "expected_column_schema": bool(schema_valid),
        "target_has_no_missing_values": bool(target.notna().all()),
        "target_is_binary_0_1": bool(
            set(target_values) == {0, 1}
            and target_numeric.notna().all()
        ),
        "no_missing_values": bool(not df.isna().any().any()),
    }

    # Keep the original CSV unchanged; save standardized column names separately.
    df.to_csv(PROCESSED_FILE, index=False)

    report = {
        "dataset_name": "Default of Credit Card Clients",
        "source": (
            "UCI Machine Learning Repository: "
            "https://archive.ics.uci.edu/dataset/350/default"
        ),
        "raw_file": str(RAW_FILE.relative_to(ROOT)),
        "standardized_file": str(PROCESSED_FILE.relative_to(ROOT)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values_by_column": {
            column: int(count)
            for column, count in missing_by_column.items()
        },
        "total_missing_values": int(missing_by_column.sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_removed": 0,
        "target_column": TARGET_COLUMN,
        "target_values_observed": [
            int(value) for value in target_values
        ],
        "target_distribution": {
            str(key): int(value)
            for key, value in target.value_counts(
                dropna=False
            ).sort_index().items()
        },
        "default_rate_percent": (
            round(float(target_numeric.mean() * 100), 2)
            if target_numeric.notna().all()
            else None
        ),
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "notes": [
            "The raw CSV was not modified.",
            "Generic feature names were standardized in the processed CSV.",
            "Duplicate rows were reported but not removed.",
            "The dataset does not contain the original customer ID column.",
        ],
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n--- DATASET VALIDATION SUMMARY ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Exact duplicate rows: {duplicate_rows}")
    print(f"Default rate: {report['default_rate_percent']}%")

    print("\n--- VALIDATION CHECKS ---")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"\nStandardized dataset: {PROCESSED_FILE}")
    print(f"Validation report: {REPORT_FILE}")
    print(f"All checks passed: {report['all_checks_passed']}")

    if not report["all_checks_passed"]:
        print(
            "\nWARNING: Some checks failed. "
            "Review the dataset and report before modeling."
        )


if __name__ == "__main__":
    main()