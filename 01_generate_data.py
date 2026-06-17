"""
01_generate_data.py
--------------------
Generates a realistic synthetic procurement dataset for the Kraljic Matrix
classification project.

Why synthetic data?
Real procurement/ERP data (SAP Ariba, Coupa, Oracle) is proprietary and never
publicly shareable. To build a portfolio project end-to-end, we simulate a
dataset that mirrors the structure, feature relationships, and *messiness*
(missing values, duplicates, inconsistent text) of a real procurement extract,
so the downstream cleaning/feature-engineering/modeling steps are genuine
work rather than a toy exercise.

The Kraljic Matrix (Peter Kraljic, HBR 1983) segments purchased items into
4 quadrants based on two axes:
    - Supply Risk      (availability, number of suppliers, lead time, etc.)
    - Profit Impact     (spend value, importance to the business)

    Quadrant        | Supply Risk | Profit Impact
    ---------------------------------------------
    Strategic       | High        | High
    Bottleneck      | High        | Low
    Leverage        | Low         | High
    Non-critical    | Low         | Low

We generate the underlying *raw business features* first, derive risk/impact
scores from them with added noise (so the relationship is realistic, not
deterministic), and then assign quadrant labels. The model is trained on the
raw business features only (not on the risk/impact scores directly) -- this
is what makes the classification a genuine ML problem rather than a lookup.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_RECORDS = 5200

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Reference lists (categories, regions, contract types)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Raw Materials": ["Steel Coil", "Aluminum Ingot", "Copper Wire", "Resin Pellets",
                       "Rare Earth Magnets", "Industrial Chemicals", "Crude Rubber"],
    "Electronics": ["Microcontrollers", "PCB Boards", "Capacitors", "Semiconductors",
                     "Sensors", "LED Modules"],
    "Packaging": ["Corrugated Boxes", "Plastic Pallets", "Shrink Wrap", "Labels",
                  "Foam Inserts"],
    "MRO": ["Industrial Bearings", "Hydraulic Hoses", "Lubricants", "Safety Gloves",
            "Spare Motors"],
    "IT Hardware": ["Laptops", "Servers", "Network Switches", "Monitors", "Storage Drives"],
    "Office Supplies": ["Printer Paper", "Stationery", "Toner Cartridges", "Desk Furniture"],
    "Logistics Services": ["Freight Forwarding", "Warehousing", "Customs Brokerage"],
    "Professional Services": ["IT Consulting", "Legal Services", "Audit Services"],
    "Chemicals": ["Specialty Adhesives", "Industrial Solvents", "Catalysts"],
    "Construction Materials": ["Cement", "Structural Steel", "Insulation Panels"],
}

REGIONS = ["North America", "Europe", "East Asia", "South Asia",
           "Latin America", "Middle East", "Africa"]

CONTRACT_TYPES = ["Spot", "Short-Term", "Long-Term", "Framework Agreement"]

# Categories more prone to single-sourcing / geopolitical exposure (raw materials,
# rare-earth-dependent electronics) vs. commoditized ones (office supplies, packaging)
HIGH_RISK_CATEGORIES = {"Raw Materials", "Electronics", "Chemicals", "Construction Materials"}
HIGH_VALUE_CATEGORIES = {"Raw Materials", "Electronics", "IT Hardware", "Professional Services"}

SUPPLIERS = [f"Supplier_{i:03d}" for i in range(1, 251)]


def generate_raw_dataframe(n=N_RECORDS) -> pd.DataFrame:
    rows = []
    for i in range(n):
        category = rng.choice(list(CATEGORIES.keys()))
        item_name = rng.choice(CATEGORIES[category])
        is_high_risk_cat = category in HIGH_RISK_CATEGORIES
        is_high_value_cat = category in HIGH_VALUE_CATEGORIES

        # --- Supply-risk-driving features ---
        num_qualified_suppliers = max(1, int(rng.gamma(3.2, 3.2) if not is_high_risk_cat
                                              else rng.gamma(1.0, 1.4)))
        avg_lead_time_days = max(1, int(rng.normal(58 if is_high_risk_cat else 13, 8)))
        single_source_flag = int(rng.random() < (0.45 if is_high_risk_cat else 0.03))
        geopolitical_risk_index = float(np.clip(
            rng.normal(72 if is_high_risk_cat else 16, 9), 0, 100))
        market_volatility_index = float(np.clip(
            rng.normal(70 if is_high_risk_cat else 16, 10), 0, 100))
        quality_defect_rate_pct = float(np.clip(rng.exponential(1.5), 0, 15))

        # --- Profit-impact-driving features ---
        base_spend = rng.lognormal(mean=12.1 if is_high_value_cat else 9.2, sigma=0.65)
        annual_spend_usd = float(round(base_spend, 2))
        pct_of_total_category_spend = float(np.clip(
            rng.beta(3, 7) * (2.6 if is_high_value_cat else 0.7), 0.001, 1))
        switching_cost_score = int(np.clip(round(rng.normal(8 if is_high_value_cat else 2.2, 1.3)), 1, 10))

        region = rng.choice(REGIONS)
        contract_type = rng.choice(CONTRACT_TYPES)
        supplier_name = rng.choice(SUPPLIERS)

        rows.append({
            "item_id": f"ITM-{i+10000}",
            "item_name": item_name,
            "category": category,
            "supplier_name": supplier_name,
            "region": region,
            "contract_type": contract_type,
            "annual_spend_usd": annual_spend_usd,
            "num_qualified_suppliers": num_qualified_suppliers,
            "avg_lead_time_days": avg_lead_time_days,
            "single_source_flag": single_source_flag,
            "geopolitical_risk_index": round(geopolitical_risk_index, 2),
            "market_volatility_index": round(market_volatility_index, 2),
            "quality_defect_rate_pct": round(quality_defect_rate_pct, 2),
            "pct_of_total_category_spend": round(pct_of_total_category_spend, 4),
            "switching_cost_score": switching_cost_score,
        })

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # 2. Derive latent Supply Risk / Profit Impact scores (with noise)
    # -----------------------------------------------------------------
    risk_z = (
        0.30 * _zscore(-df["num_qualified_suppliers"]) +
        0.20 * _zscore(df["avg_lead_time_days"]) +
        0.20 * _zscore(df["single_source_flag"]) +
        0.15 * _zscore(df["geopolitical_risk_index"]) +
        0.15 * _zscore(df["market_volatility_index"])
    )
    impact_z = (
        0.45 * _zscore(df["annual_spend_usd"]) +
        0.30 * _zscore(df["pct_of_total_category_spend"]) +
        0.25 * _zscore(df["switching_cost_score"])
    )

    # Add realistic measurement noise so labels aren't a perfectly deterministic
    # function of features (in practice Kraljic scores include some analyst
    # judgment on top of the quantitative inputs) -- but keep it modest, since
    # the score IS computed primarily from these same procurement KPIs.
    risk_z = risk_z + rng.normal(0, 0.02, size=len(df))
    impact_z = impact_z + rng.normal(0, 0.02, size=len(df))

    risk_med, impact_med = np.median(risk_z), np.median(impact_z)
    high_risk = risk_z > risk_med
    high_impact = impact_z > impact_med

    def label(hr, hi):
        if hr and hi:
            return "Strategic"
        if hr and not hi:
            return "Bottleneck"
        if not hr and hi:
            return "Leverage"
        return "Non-Critical"

    df["kraljic_quadrant"] = [label(hr, hi) for hr, hi in zip(high_risk, high_impact)]

    # -----------------------------------------------------------------
    # 3. Inject realistic "dirty data" so cleaning step is meaningful
    # -----------------------------------------------------------------
    # a) Missing values in a few columns (MCAR-ish)
    for col, frac in [("avg_lead_time_days", 0.025), ("quality_defect_rate_pct", 0.02),
                       ("region", 0.01), ("geopolitical_risk_index", 0.015)]:
        idx = rng.choice(df.index, size=int(frac * len(df)), replace=False)
        df.loc[idx, col] = np.nan

    # b) Inconsistent text casing / whitespace for categorical text fields
    def messy_text(x):
        roll = rng.random()
        if roll < 0.15:
            return x.upper()
        if roll < 0.30:
            return x.lower()
        if roll < 0.35:
            return f" {x} "
        return x

    df["category"] = df["category"].apply(messy_text)
    df["region"] = df["region"].apply(lambda x: messy_text(x) if pd.notna(x) else x)

    # c) Duplicate rows (simulating double-loaded ERP extracts)
    dup_idx = rng.choice(df.index, size=60, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # d) A few extreme outliers in spend (data entry errors, e.g. missing decimal)
    out_idx = rng.choice(df.index, size=8, replace=False)
    df.loc[out_idx, "annual_spend_usd"] = df.loc[out_idx, "annual_spend_usd"] * 100

    # e) Shuffle rows so it doesn't look "generated in order"
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.Series(s)
    return (s - s.mean()) / (s.std() + 1e-9)


if __name__ == "__main__":
    df = generate_raw_dataframe()
    out_path = "/home/claude/kraljic_project/data/procurement_data_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass distribution (target leakage check - should be roughly balanced):")
    print(df["kraljic_quadrant"].value_counts())
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print(f"\nDuplicate rows: {df.duplicated(subset=['item_id']).sum()}")
