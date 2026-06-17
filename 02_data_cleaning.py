"""
02_data_cleaning.py
--------------------
Cleans the raw procurement extract:
  1. Removes duplicate records (by item_id)
  2. Standardizes inconsistent text (casing, whitespace) in categorical columns
  3. Handles missing values:
       - numeric columns -> median imputation (robust to outliers, simple & defensible)
       - categorical columns -> mode imputation
  4. Caps extreme outliers in annual_spend_usd using the IQR rule (data-entry errors)
  5. Enforces correct dtypes
  6. Saves a clean dataset + a short data-quality report
"""

import numpy as np
import pandas as pd

RAW_PATH = "/home/claude/kraljic_project/data/procurement_data_raw.csv"
CLEAN_PATH = "/home/claude/kraljic_project/data/procurement_data_clean.csv"
REPORT_PATH = "/home/claude/kraljic_project/outputs/data_quality_report.txt"

NUMERIC_COLS = [
    "annual_spend_usd", "num_qualified_suppliers", "avg_lead_time_days",
    "geopolitical_risk_index", "market_volatility_index", "quality_defect_rate_pct",
    "pct_of_total_category_spend", "switching_cost_score",
]
CATEGORICAL_COLS = ["category", "region", "contract_type", "supplier_name"]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    report = []
    report.append(f"Raw rows: {len(df)}")

    # 1. Drop exact duplicates on business key
    before = len(df)
    df = df.drop_duplicates(subset=["item_id"]).reset_index(drop=True)
    report.append(f"Removed {before - len(df)} duplicate rows (duplicate item_id)")

    # 2. Standardize text columns: strip whitespace, title-case
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace("Nan", np.nan)
    report.append(f"Standardized text formatting in: {CATEGORICAL_COLS}")

    # 3. Missing value handling
    missing_before = df.isna().sum()
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)
    report.append("Missing values imputed (median for numeric, mode for categorical):")
    for col, n in missing_before.items():
        if n > 0:
            report.append(f"  - {col}: {n} missing -> imputed")

    # 4. Outlier capping via IQR rule on spend
    q1, q3 = df["annual_spend_usd"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr  # wide bound (3x IQR) to only catch true data-entry errors
    n_capped = int((df["annual_spend_usd"] > upper_bound).sum())
    df["annual_spend_usd"] = np.where(
        df["annual_spend_usd"] > upper_bound, upper_bound, df["annual_spend_usd"]
    )
    report.append(f"Capped {n_capped} extreme outliers in annual_spend_usd at {upper_bound:,.0f} (3x IQR rule)")

    # 5. Dtype enforcement
    df["num_qualified_suppliers"] = df["num_qualified_suppliers"].astype(int)
    df["single_source_flag"] = df["single_source_flag"].astype(int)
    df["switching_cost_score"] = df["switching_cost_score"].astype(int)

    report.append(f"Final clean rows: {len(df)}")
    report.append(f"Remaining nulls: {int(df.isna().sum().sum())}")

    return df, report


if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_PATH)
    clean_df, report_lines = clean(raw_df)
    clean_df.to_csv(CLEAN_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        f.write("DATA QUALITY / CLEANING REPORT\n")
        f.write("=" * 40 + "\n")
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nClean data saved -> {CLEAN_PATH}")
    print(f"Report saved -> {REPORT_PATH}")
