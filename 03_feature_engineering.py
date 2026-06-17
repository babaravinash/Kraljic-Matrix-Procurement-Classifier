"""
03_feature_engineering.py
--------------------------
Builds the model-ready feature matrix from the cleaned dataset.

New engineered features (business-driven, not just raw columns):
  - spend_per_supplier        : annual_spend_usd / num_qualified_suppliers
                                 (concentration of spend -> proxy for leverage)
  - log_annual_spend          : log1p transform to tame right-skewed spend distribution
  - supplier_scarcity_score   : inverse of num_qualified_suppliers (bounded 0-1)
  - lead_time_bucket          : binned lead time (Fast / Medium / Slow) for interpretability
  - risk_exposure_index       : composite of geopolitical + market volatility (0-100)
  - high_value_category_flag  : 1 if category historically carries high spend concentration

Categorical encoding:
  - One-hot encoding for category, region, contract_type
  - supplier_name is dropped as a direct feature (250 suppliers, high cardinality,
    not meaningfully predictive on its own -- would just encourage overfitting /
    memorization rather than learning transferable supply-risk patterns)

Target encoding:
  - LabelEncoder on kraljic_quadrant (Strategic / Leverage / Bottleneck / Non-Critical)
"""

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CLEAN_PATH = "/home/claude/kraljic_project/data/procurement_data_clean.csv"
FEATURES_PATH = "/home/claude/kraljic_project/data/procurement_data_features.csv"
ENCODERS_PATH = "/home/claude/kraljic_project/models/label_encoder_classes.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Derived numeric features ---
    df["spend_per_supplier"] = df["annual_spend_usd"] / df["num_qualified_suppliers"]
    df["log_annual_spend"] = np.log1p(df["annual_spend_usd"])
    df["supplier_scarcity_score"] = 1 / df["num_qualified_suppliers"]
    df["risk_exposure_index"] = (
        0.5 * df["geopolitical_risk_index"] + 0.5 * df["market_volatility_index"]
    )

    def lead_time_bucket(days):
        if days <= 15:
            return "Fast"
        if days <= 35:
            return "Medium"
        return "Slow"

    df["lead_time_bucket"] = df["avg_lead_time_days"].apply(lead_time_bucket)

    high_value_cats = {"Raw Materials", "Electronics", "It Hardware", "Professional Services"}
    df["high_value_category_flag"] = df["category"].isin(high_value_cats).astype(int)

    # --- One-hot encode low-cardinality categoricals ---
    df = pd.get_dummies(
        df,
        columns=["category", "region", "contract_type", "lead_time_bucket"],
        prefix=["cat", "region", "contract", "leadtime"],
    )

    # --- Drop columns not used as model features ---
    df = df.drop(columns=["item_id", "item_name", "supplier_name"])

    # --- Encode target ---
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["kraljic_quadrant"])
    df = df.drop(columns=["kraljic_quadrant"])

    with open(ENCODERS_PATH, "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f, indent=2)

    return df


if __name__ == "__main__":
    clean_df = pd.read_csv(CLEAN_PATH)
    feat_df = engineer_features(clean_df)
    feat_df.to_csv(FEATURES_PATH, index=False)

    print(f"Feature matrix shape: {feat_df.shape}")
    print(f"Columns: {list(feat_df.columns)}")
    print(f"\nSaved -> {FEATURES_PATH}")
    print(f"Label encoder classes saved -> {ENCODERS_PATH}")
