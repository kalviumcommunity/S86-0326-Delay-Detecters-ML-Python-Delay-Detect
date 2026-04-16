"""Lesson 5.33 demo: interpreting feature importance from tree-based models.

Run:
    python scripts/feature_importance_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.feature_importance_analysis import (
    aggregate_importance_to_raw_features,
    build_random_forest_classifier_pipeline,
    compare_importance_tables,
    evaluate_drop_impact,
    find_high_correlation_pairs,
    get_mdi_importance_table,
    get_permutation_importance_table,
    train_random_forest_pipeline,
)


def save_bar_plot(df, feature_col: str, value_col: str, title: str, out_path: Path) -> None:
    top = df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top[feature_col], top[value_col], color="steelblue", edgecolor="white")
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_correlation_heatmap(X_numeric, out_path: Path) -> None:
    corr = X_numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix (Numeric Features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    print("=== Feature Importance Demo (Tree-Based Models) ===")

    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    model = build_random_forest_classifier_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=Config.RANDOM_STATE,
    )
    model = train_random_forest_pipeline(model, X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Train/test gap: {train_acc - test_acc:.4f}")

    mdi_df = get_mdi_importance_table(model)
    perm_df = get_permutation_importance_table(
        model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=25,
        random_state=Config.RANDOM_STATE,
    )
    cmp_df = compare_importance_tables(mdi_df, perm_df)

    print("\nTop MDI importance features:")
    print(mdi_df.head(12).to_string(index=False))

    print("\nTop permutation-importance features:")
    print(perm_df.head(12).to_string(index=False))

    print("\nLargest MDI vs permutation ranking disagreements:")
    rank_shift = cmp_df.reindex(cmp_df["Rank_Delta_MDI_minus_Permutation"].abs().sort_values(ascending=False).index)
    print(rank_shift.head(10).to_string(index=False))

    numeric_corr_pairs = find_high_correlation_pairs(X_train[Config.NUMERICAL_COLS], threshold=0.8)
    print("\nHighly correlated numeric feature pairs (|corr| >= 0.80):")
    if numeric_corr_pairs.empty:
        print("  None found")
    else:
        print(numeric_corr_pairs.to_string(index=False))

    # Raw-feature drop candidates: negative or near-zero permutation with near-zero MDI.
    raw_perm = aggregate_importance_to_raw_features(
        perm_df,
        importance_col="Permutation_Importance",
        std_col="Permutation_Std",
    )
    raw_mdi = aggregate_importance_to_raw_features(
        mdi_df,
        importance_col="MDI_Importance",
        std_col="MDI_Std",
    )
    raw_joined = raw_mdi.merge(raw_perm, on="Feature", how="inner")

    candidates = raw_joined[
        (raw_joined["Permutation_Importance"] <= 0.0)
        | ((raw_joined["Permutation_Importance"] < 0.005) & (raw_joined["MDI_Importance"] < 0.02))
    ]["Feature"].tolist()
    candidates = [f for f in candidates if f in (Config.NUMERICAL_COLS + Config.CATEGORICAL_COLS)]

    if candidates:
        impact = evaluate_drop_impact(X_train, X_test, y_train, y_test, candidates)
        print("\nDrop-candidate retrain check:")
        print(f"  Dropped features: {impact['dropped_features']}")
        print(f"  Baseline accuracy: {impact['baseline_accuracy']:.4f}")
        print(f"  Dropped accuracy:  {impact['dropped_accuracy']:.4f}")
        print(f"  Accuracy delta:    {impact['accuracy_delta']:+.4f}")
    else:
        print("\nNo low-value raw-feature candidates met the drop criteria.")

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_bar_plot(
        mdi_df,
        feature_col="Feature",
        value_col="MDI_Importance",
        title="Feature Importance (MDI)",
        out_path=out_dir / "lesson_5_33_mdi_importance.png",
    )
    save_bar_plot(
        perm_df,
        feature_col="Feature",
        value_col="Permutation_Importance",
        title="Feature Importance (Permutation)",
        out_path=out_dir / "lesson_5_33_permutation_importance.png",
    )
    save_correlation_heatmap(
        X_train[Config.NUMERICAL_COLS],
        out_path=out_dir / "lesson_5_33_correlation_heatmap.png",
    )

    print("\nSaved figures:")
    print("  reports/lesson_5_33_mdi_importance.png")
    print("  reports/lesson_5_33_permutation_importance.png")
    print("  reports/lesson_5_33_correlation_heatmap.png")


if __name__ == "__main__":
    main()
