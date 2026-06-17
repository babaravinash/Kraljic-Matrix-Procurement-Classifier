"""
04_train_model.py
------------------
Trains a Random Forest classifier to predict the Kraljic Matrix quadrant.

Steps:
  1. Stratified train/test split (80/20) -- stratified because we care about
     balanced performance across all 4 quadrants, not just majority class accuracy.
  2. Baseline Random Forest (default params) for comparison.
  3. Hyperparameter tuning via GridSearchCV (5-fold cross-validation) over
     n_estimators, max_depth, min_samples_split, min_samples_leaf.
  4. Refit best estimator on full training set.
  5. Persist the trained model + feature column order (needed for inference).
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

FEATURES_PATH = "/home/claude/kraljic_project/data/procurement_data_features.csv"
MODEL_PATH = "/home/claude/kraljic_project/models/random_forest_kraljic_model.pkl"
COLUMNS_PATH = "/home/claude/kraljic_project/models/feature_columns.json"
CV_RESULTS_PATH = "/home/claude/kraljic_project/outputs/cv_results.csv"
SPLIT_PATH = "/home/claude/kraljic_project/data/"

RANDOM_SEED = 42


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Persist test split so the evaluation script uses the exact same holdout
    X_train.to_csv(f"{SPLIT_PATH}X_train.csv", index=False)
    X_test.to_csv(f"{SPLIT_PATH}X_test.csv", index=False)
    y_train.to_csv(f"{SPLIT_PATH}y_train.csv", index=False)
    y_test.to_csv(f"{SPLIT_PATH}y_test.csv", index=False)

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # --- Baseline model (untuned) for comparison ---
    baseline = RandomForestClassifier(random_state=RANDOM_SEED)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline RandomForest (default params) test accuracy: {baseline_acc:.4f}")

    # --- Hyperparameter tuning ---
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    print("\nRunning GridSearchCV (4-fold)... this may take a minute.")
    t0 = time.time()
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        param_grid=param_grid,
        cv=4,
        scoring="f1_macro",
        n_jobs=1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\nGridSearch completed in {elapsed:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV f1_macro score: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    tuned_acc = accuracy_score(y_test, best_model.predict(X_test))
    tuned_f1 = f1_score(y_test, best_model.predict(X_test), average="macro")
    print(f"\nTuned model test accuracy: {tuned_acc:.4f}")
    print(f"Tuned model test f1_macro: {tuned_f1:.4f}")
    print(f"Improvement over baseline: {(tuned_acc - baseline_acc) * 100:.2f} percentage points")

    # Save cross-validation results for transparency / interview talking points
    cv_results_df = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    cv_results_df.to_csv(CV_RESULTS_PATH, index=False)

    # Persist model + feature column order
    joblib.dump(best_model, MODEL_PATH)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    summary = {
        "baseline_accuracy": round(baseline_acc, 4),
        "tuned_accuracy": round(tuned_acc, 4),
        "tuned_f1_macro": round(tuned_f1, 4),
        "best_params": grid.best_params_,
        "best_cv_f1_macro": round(grid.best_score_, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open("/home/claude/kraljic_project/outputs/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModel saved -> {MODEL_PATH}")
    print(f"Feature columns saved -> {COLUMNS_PATH}")
    print(f"Training summary saved -> /home/claude/kraljic_project/outputs/training_summary.json")


if __name__ == "__main__":
    main()
