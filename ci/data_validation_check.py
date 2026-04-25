from __future__ import annotations

import argparse
import sys

import pandas as pd


REQUIRED_COLUMNS = {"TransactionID", "TransactionDT", "isFraud"}


def validate(path: str) -> int:
    df = pd.read_csv(path, nrows=5000)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        print(f"Missing required columns: {sorted(missing_columns)}")
        return 1

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    severe_missing = missing_ratio[missing_ratio > 0.50]
    if not severe_missing.empty:
        print("Columns with >50% missing values:")
        print(severe_missing)

    print("Data validation checks completed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    sys.exit(validate(args.data))


if __name__ == "__main__":
    main()
