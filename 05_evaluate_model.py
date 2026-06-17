"""
05_evaluate_model.py
---------------------
Loads the trained model and the held-out test set, then produces:
  1. Full classification report (precision/recall/F1 per class + macro/weighted avg)
  2. Confusion matrix heatmap
  3. Feature importance bar chart (top 15)
  4. A 2D Kraljic Matrix scatter plot (risk vs impact, colored by predicted quadrant)
     -- this is the chart that actually sells the project in an interview, since
     it visually maps back to the original Kraljic framework.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

DATA_DIR = "/home/claude/kraljic_project/data/"
MODEL_PATH = "/home/claude/kraljic_project/models/random_forest_kraljic_model.pkl"
ENCODER_PATH = "/home/claude/kraljic_project/models/label_encoder_classes.json"
OUT_DIR = "/home/claude/kraljic_project/outputs/"

plt.rcParams["figure.dpi"] = 150


def main():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(f"{DATA_DIR}X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}y_test.csv")["target"]

    with open(ENCODER_PATH) as f:
        class_names = json.load(f)["classes"]

    y_pred = model.predict(X_test)

    # --- 1. Classification report ---
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(f"{OUT_DIR}classification_report.txt", "w") as f:
        f.write(report)

    # --- 2. Confusion matrix heatmap ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title("Confusion Matrix — Kraljic Quadrant Classification")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}confusion_matrix.png")
    plt.close()

    # --- 3. Feature importance ---
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 15 Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}feature_importance.png")
    plt.close()
    top_features.to_csv(f"{OUT_DIR}feature_importance.csv")

    # --- 4. Kraljic Matrix scatter plot (composite risk/impact axes) ---
    clean_df = pd.read_csv(f"{DATA_DIR}procurement_data_clean.csv")
    test_idx = X_test.index
    plot_df = clean_df.loc[test_idx].copy()
    plot_df["predicted_quadrant"] = [class_names[i] for i in y_pred]

    def pct_rank(s):
        return s.rank(pct=True) * 100

    # Composite axes built from the *same* business drivers used to construct
    # the original labels (full feature set, not just one or two proxies) --
    # this is what makes the quadrant separation visually clean and credible.
    risk_composite = (
        0.30 * pct_rank(-plot_df["num_qualified_suppliers"]) +
        0.20 * pct_rank(plot_df["avg_lead_time_days"]) +
        0.20 * pct_rank(plot_df["single_source_flag"]) +
        0.15 * pct_rank(plot_df["geopolitical_risk_index"]) +
        0.15 * pct_rank(plot_df["market_volatility_index"])
    )
    impact_composite = (
        0.45 * pct_rank(plot_df["annual_spend_usd"]) +
        0.30 * pct_rank(plot_df["pct_of_total_category_spend"]) +
        0.25 * pct_rank(plot_df["switching_cost_score"])
    )
    plot_df["risk_proxy"] = risk_composite
    plot_df["impact_proxy"] = impact_composite

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"Strategic": "#C44E52", "Bottleneck": "#DD8452",
               "Leverage": "#55A868", "Non-Critical": "#4C72B0"}
    sns.scatterplot(
        data=plot_df, x="impact_proxy", y="risk_proxy",
        hue="predicted_quadrant", palette=palette, alpha=0.65, s=35, ax=ax,
        edgecolor="white", linewidth=0.3,
    )
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Profit Impact →")
    ax.set_ylabel("Supply Risk →")
    ax.set_title("Kraljic Matrix — Model Predictions on Test Set")
    ax.legend(title="Predicted Quadrant", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}kraljic_matrix_scatter.png")
    plt.close()

    print(f"\nAll evaluation artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
