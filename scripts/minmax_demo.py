"""MinMax normalization demo for Lesson 5.20.

Shows:
1) Leakage-safe train/test scaling with MinMaxScaler.
2) ColumnTransformer pipeline with numerical + categorical processing.
3) Outlier sensitivity illustration.
4) Saving and loading a full preprocessing+model pipeline.

Run:
    python scripts/minmax_demo.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


RANDOM_STATE = 42


def make_dataset(n_samples: int = 1200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a synthetic churn-like dataset with mixed feature types."""
    rng = np.random.RandomState(seed)

    tenure = rng.randint(1, 72, size=n_samples)
    monthly_charges = rng.normal(loc=70, scale=25, size=n_samples).clip(10, 180)
    total_charges = (tenure * monthly_charges + rng.normal(0, 100, size=n_samples)).clip(0)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.58, 0.24, 0.18])

    # Create a non-trivial classification target.
    risk_signal = (
        0.04 * (monthly_charges - 70)
        - 0.03 * (tenure - 24)
        + 0.0008 * (total_charges - 2200)
        + (contract == "Month-to-month") * 0.6
        + rng.normal(0, 1.0, size=n_samples)
    )
    churn = (risk_signal > 0.35).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "Churn": churn,
        }
    )


def build_pipeline(numerical_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Build leakage-safe preprocessing+model pipeline with MinMaxScaler."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def demonstrate_outlier_effect() -> None:
    """Show how a single outlier compresses most MinMax values."""
    balances = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 50000], dtype=float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(balances.reshape(-1, 1)).ravel()

    print("\nOutlier sensitivity demo:")
    print("Original balances:", balances.tolist())
    print("Scaled balances:", [round(v, 4) for v in scaled.tolist()])
    print("Notice how non-outlier values are squeezed near 0.")


def main() -> None:
    df = make_dataset()

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = ["Contract"]
    target_col = "Churn"

    X = df[numerical_cols + categorical_cols]
    y = df[target_col]

    # Split first to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit model pipeline on train only.
    pipeline = build_pipeline(numerical_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    # Evaluate on test set.
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Inspect fitted numerical scaler ranges.
    fitted_scaler = pipeline.named_steps["preprocessing"].named_transformers_["num"]
    print("Training min values:", dict(zip(numerical_cols, fitted_scaler.data_min_)))
    print("Training max values:", dict(zip(numerical_cols, fitted_scaler.data_max_)))

    X_train_scaled = fitted_scaler.transform(X_train[numerical_cols])
    train_min = X_train_scaled.min(axis=0)
    train_max = X_train_scaled.max(axis=0)

    print("Scaled training mins:", dict(zip(numerical_cols, np.round(train_min, 4))))
    print("Scaled training maxs:", dict(zip(numerical_cols, np.round(train_max, 4))))
    print(f"Test accuracy: {acc:.3f}")
    print(f"Test F1 score: {f1:.3f}")

    # Save and reload full pipeline artifact.
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "minmax_logreg_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    loaded = joblib.load(pipeline_path)
    reloaded_pred = loaded.predict(X_test.head(5))
    print(f"Saved pipeline to: {pipeline_path}")
    print("Sample predictions from reloaded pipeline:", reloaded_pred.tolist())

    demonstrate_outlier_effect()


if __name__ == "__main__":
    main()
