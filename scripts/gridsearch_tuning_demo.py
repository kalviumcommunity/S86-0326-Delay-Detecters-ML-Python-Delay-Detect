"""Lesson 5.34 demo: improving model performance using GridSearchCV.

Run:
    python scripts/gridsearch_tuning_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.hyperparameter_tuning import (
    build_knn_pipeline,
    evaluate_best_on_test,
    run_knn_grid_search,
    run_knn_random_search,
    summarize_search_results,
)


def save_knn_search_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    """Save KNN train/CV trend and CV uncertainty over K."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for w, style in [("uniform", "o-"), ("distance", "s:")]:
        mask = results_df["param_knn__weights"] == w
        if not mask.any():
            continue

        x = results_df.loc[mask, "param_knn__n_neighbors"].astype(int)
        y_cv = results_df.loc[mask, "mean_test_score"].astype(float)
        y_std = results_df.loc[mask, "std_test_score"].astype(float)

        ax.plot(x, y_cv, style, label=f"CV ({w})", alpha=0.9)
        ax.fill_between(x, y_cv - y_std, y_cv + y_std, alpha=0.12)

    # Train curve for one setting for leakage/overfit diagnosis.
    train_mask = results_df["param_knn__weights"] == "uniform"
    if train_mask.any() and "mean_train_score" in results_df.columns:
        ax.plot(
            results_df.loc[train_mask, "param_knn__n_neighbors"].astype(int),
            results_df.loc[train_mask, "mean_train_score"].astype(float),
            "--",
            label="Train (uniform)",
            alpha=0.8,
        )

    ax.set_xlabel("K (n_neighbors)")
    ax.set_ylabel("Score")
    ax.set_title("GridSearchCV: KNN Hyperparameter Search")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    print("=== GridSearchCV Tuning Demo (Lesson 5.34) ===")

    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    # Baseline and untuned benchmarks for honest tuning context.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)

    untuned = build_knn_pipeline(X_train)
    untuned.set_params(knn__n_neighbors=5, knn__weights="uniform")
    untuned.fit(X_train, y_train)
    untuned_acc = untuned.score(X_test, y_test)

    print(f"Baseline test accuracy: {baseline_acc:.4f}")
    print(f"Untuned test accuracy:  {untuned_acc:.4f}")

    grid = run_knn_grid_search(X_train, y_train, scoring="f1", cv=5)

    print(f"\nBest GridSearch parameters: {grid.best_params_}")
    print(f"Best GridSearch CV F1:      {grid.best_score_:.4f}")

    summary = summarize_search_results(grid, top_n=10)
    print("\nTop GridSearch configurations:")
    print(summary.to_string(index=False))

    final_eval = evaluate_best_on_test(grid, X_test, y_test)
    print(f"\nFinal test {final_eval['score_name']}: {final_eval['test_score']:.4f}")
    print("\nClassification report (best estimator on test set):")
    print(final_eval["classification_report"])

    # Randomized search comparison for computational trade-off discussion.
    random_search = run_knn_random_search(X_train, y_train, scoring="f1", cv=5, n_iter=25)
    print(f"RandomizedSearch best params: {random_search.best_params_}")
    print(f"RandomizedSearch best CV F1:  {random_search.best_score_:.4f}")

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_results = pd.DataFrame(grid.cv_results_).sort_values("param_knn__n_neighbors")
    save_knn_search_plot(full_results, out_dir / "lesson_5_34_gridsearch_knn_curve.png")

    summary.to_csv(out_dir / "lesson_5_34_gridsearch_top10.csv", index=False)

    print("\nSaved artifacts:")
    print("  reports/lesson_5_34_gridsearch_knn_curve.png")
    print("  reports/lesson_5_34_gridsearch_top10.csv")


if __name__ == "__main__":
    main()
