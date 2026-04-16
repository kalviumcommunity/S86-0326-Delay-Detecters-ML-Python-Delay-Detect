"""Lesson 5.32 demo: training and interpreting decision tree models.

Run:
    python scripts/decision_tree_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_preprocessing import clean_data, load_data, split_data
from src.decision_tree_modeling import (
    build_decision_tree_classifier_pipeline,
    build_decision_tree_regressor,
    build_feature_importance_frame,
    compare_classification_accuracy,
    cross_validate_decision_tree_classifier,
    cross_validate_decision_tree_regression,
    evaluate_decision_tree_classification,
    evaluate_decision_tree_regression,
    get_tree_feature_names,
    tune_decision_tree_classifier_depth,
)


def classification_demo() -> None:
    print("=== Decision Tree Classification Demo (Project Data) ===")
    df = load_data(Config.RAW_DATA_PATH)
    df = clean_data(df, target_column=Config.TARGET_COLUMN)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=Config.TARGET_COLUMN,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
    )

    baseline = DummyClassifier(strategy="most_frequent", random_state=Config.RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    tree = build_decision_tree_classifier_pipeline(
        categorical_cols=Config.CATEGORICAL_COLS,
        numerical_cols=Config.NUMERICAL_COLS,
        criterion="gini",
        max_depth=4,
        min_samples_leaf=5,
        random_state=Config.RANDOM_STATE,
    )

    depth_grid = list(range(1, 13))
    tuning = tune_decision_tree_classifier_depth(
        tree,
        X_train,
        y_train,
        depth_values=depth_grid,
        cv=5,
        random_state=Config.RANDOM_STATE,
    )

    best_tree = tree.set_params(model__max_depth=tuning["best_depth"])
    best_tree.fit(X_train, y_train)

    y_pred = best_tree.predict(X_test)
    train_acc = best_tree.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    metrics = evaluate_decision_tree_classification(y_test, y_pred)
    gain = compare_classification_accuracy(test_acc, baseline_acc)

    print(f"Baseline accuracy:     {baseline_acc:.4f}")
    print(f"Tree train accuracy:   {train_acc:.4f}")
    print(f"Tree test accuracy:    {test_acc:.4f}")
    print(f"Train/test gap:        {train_acc - test_acc:.4f}")
    print(f"Accuracy gain vs base: {gain['accuracy_gain']:.4f}")
    print("Tree metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    cv_summary = cross_validate_decision_tree_classifier(best_tree, X_train, y_train, cv=5, random_state=Config.RANDOM_STATE)
    print("CV accuracy scores:", np.round(cv_summary["accuracy_scores"], 4).tolist())
    print(f"Mean CV accuracy: {cv_summary['accuracy_mean']:.4f} +/- {cv_summary['accuracy_std']:.4f}")
    print("CV F1 scores:", np.round(cv_summary["f1_scores"], 4).tolist())
    print(f"Mean CV F1: {cv_summary['f1_mean']:.4f} +/- {cv_summary['f1_std']:.4f}")

    print("\nDepth tuning (accuracy):")
    for depth, train_score, cv_score in zip(tuning["depth_values"], tuning["mean_train_scores"], tuning["mean_cv_scores"]):
        print(f"  depth={depth:2d} | train={train_score:.4f} | cv={cv_score:.4f}")
    print(f"Best depth: {tuning['best_depth']}")
    print(f"Best CV accuracy: {tuning['best_cv_score']:.4f}")

    feature_importance = build_feature_importance_frame(best_tree)
    print("\nTop feature importances:")
    print(feature_importance.head(10).to_string(index=False))

    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_fig, depth_ax = plt.subplots(figsize=(9, 5))
    depth_ax.plot(tuning["depth_values"], tuning["mean_train_scores"], marker="o", label="Train Accuracy")
    depth_ax.plot(tuning["depth_values"], tuning["mean_cv_scores"], marker="o", label="CV Accuracy")
    depth_ax.set_xlabel("Max Depth")
    depth_ax.set_ylabel("Accuracy")
    depth_ax.set_title("Decision Tree: Depth vs. Accuracy")
    depth_ax.grid(True, alpha=0.3)
    depth_ax.legend()
    depth_fig.tight_layout()
    depth_path = output_dir / "lesson_5_32_depth_curve.png"
    depth_fig.savefig(depth_path, dpi=140)
    plt.close(depth_fig)
    print(f"Saved depth curve to: {depth_path}")

    tree_fig = plt.figure(figsize=(20, 10))
    plot_tree(
        best_tree.named_steps["model"],
        feature_names=get_tree_feature_names(best_tree),
        class_names=["On Time", "Delayed"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree - Delay Prediction")
    plt.tight_layout()
    tree_path = output_dir / "lesson_5_32_tree.png"
    tree_fig.savefig(tree_path, dpi=140)
    plt.close(tree_fig)
    print(f"Saved tree visualization to: {tree_path}")


def regression_demo() -> None:
    print("\n=== Decision Tree Regression Demo (Synthetic Data) ===")
    rng = np.random.RandomState(42)
    n_samples = 260

    X = np.column_stack(
        [
            rng.uniform(0, 20, size=n_samples),
            rng.uniform(0, 1, size=n_samples),
            rng.uniform(10, 200, size=n_samples),
        ]
    )
    noise = rng.normal(0, 1.1, size=n_samples)
    y = 2.8 * np.sin(X[:, 0] / 3.2) + 1.7 * X[:, 1] + 0.025 * X[:, 2] + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=Config.RANDOM_STATE,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_decision_tree_regression(y_test, baseline_pred)

    tree_reg = build_decision_tree_regressor(
        criterion="squared_error",
        max_depth=4,
        min_samples_leaf=5,
        random_state=Config.RANDOM_STATE,
    )
    tree_reg.fit(X_train, y_train)
    y_pred = tree_reg.predict(X_test)

    metrics = evaluate_decision_tree_regression(y_test, y_pred)
    train_r2 = tree_reg.score(X_train, y_train)
    test_r2 = metrics["r2"]
    cv_summary = cross_validate_decision_tree_regression(tree_reg, X_train, y_train, cv=5, random_state=Config.RANDOM_STATE)

    print("Baseline regression metrics:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    print("Decision tree regression metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print(f"Train R2: {train_r2:.4f}")
    print(f"Train/test gap: {train_r2 - test_r2:.4f}")
    print("CV RMSE scores:", np.round(cv_summary["rmse_scores"], 4).tolist())
    print(f"Mean CV RMSE: {cv_summary['rmse_mean']:.4f} +/- {cv_summary['rmse_std']:.4f}")
    print("CV R2 scores:", np.round(cv_summary["r2_scores"], 4).tolist())
    print(f"Mean CV R2: {cv_summary['r2_mean']:.4f} +/- {cv_summary['r2_std']:.4f}")


def main() -> None:
    classification_demo()
    regression_demo()


if __name__ == "__main__":
    main()