"""Lesson 5.31 demo: bias-variance diagnosis through model behavior.

Run:
    python scripts/bias_variance_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow direct script execution from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bias_variance import diagnose_train_test_behavior, interpret_learning_curve, summarize_cv_stability


def evaluate_knn_by_k(X_train, X_test, y_train, y_test, k_values: list[int]) -> None:
    print("=== Train/Test Behavior by K ===")
    target_good_accuracy = 0.92
    for k in k_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k)),
        ])
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        diagnosis = diagnose_train_test_behavior(
            train_score=train_acc,
            test_score=test_acc,
            good_score_threshold=target_good_accuracy,
            large_gap_threshold=0.08,
        )

        print(f"K={k:3d} | train={train_acc:.4f} | test={test_acc:.4f} | gap={diagnosis['gap']:.4f}")
        print(f"  diagnosis={diagnosis['diagnosis']} | action={diagnosis['recommendation']}")


def learning_curve_demo(X_train, y_train, k: int, ax, title: str) -> None:
    target_good_accuracy = 0.92
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=k)),
    ])

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        n_jobs=-1,
    )

    interpretation = interpret_learning_curve(
        train_scores=train_scores,
        val_scores=val_scores,
        good_score_threshold=target_good_accuracy,
        large_gap_threshold=0.08,
    )

    cv_summary = summarize_cv_stability(val_scores.mean(axis=0))

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, marker="o", label="Train Accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    print(f"\n=== Learning Curve Summary: {title} ===")
    print(f"Final train accuracy: {interpretation['final_train']:.4f}")
    print(f"Final val accuracy:   {interpretation['final_val']:.4f}")
    print(f"Final gap:            {interpretation['final_gap']:.4f}")
    print(f"Interpretation:       {interpretation['diagnosis']}")
    print(f"Action:               {interpretation['recommendation']}")
    print(f"CV stability:         {cv_summary['stability']} (std={cv_summary['std']:.4f})")


def main() -> None:
    X, y = make_moons(n_samples=1200, noise=0.28, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    evaluate_knn_by_k(X_train, X_test, y_train, y_test, k_values=[1, 15, 70])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    learning_curve_demo(X_train, y_train, k=1, ax=axes[0], title="KNN (K=1) - Variance Risk")
    learning_curve_demo(X_train, y_train, k=70, ax=axes[1], title="KNN (K=70) - Bias Risk")

    output_path = Path("reports") / "lesson_5_31_learning_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    print(f"\nSaved learning-curve figure to: {output_path}")


if __name__ == "__main__":
    main()
