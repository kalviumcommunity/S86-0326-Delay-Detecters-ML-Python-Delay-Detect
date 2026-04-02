"""
Feature inspection utilities.

Provides a small runnable function to validate feature/target definitions,
separate X and y, print shapes and basic distribution summaries, and
produce simple per-column statistics to guide preprocessing decisions.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import Config
from src.data_preprocessing import validate_feature_definition


def inspect_features(raw_path: str | Path = None) -> None:
    path = Path(raw_path) if raw_path else Config.RAW_DATA_PATH
    df = pd.read_csv(path)

    target = Config.TARGET_COLUMN
    features = Config.ALL_FEATURES
    excluded = getattr(Config, "EXCLUDED_COLUMNS", [])

    # Validate definitions
    validate_feature_definition(df, target, features, excluded)

    # Separate
    X = df[features].copy()
    y = df[target].copy()

    print(f"Data loaded: {len(df)} rows")
    print(f"Features: {X.shape} | Target: {y.shape}")
    print("Target distribution:")
    print(y.value_counts(dropna=False, normalize=True))

    # Numerical summaries
    num_cols = [c for c in features if c in df.select_dtypes(include=[np.number]).columns]
    if num_cols:
        print("\nNumerical feature summaries:")
        print(df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)
        for col in num_cols:
            skew = df[col].skew()
            pct_99 = df[col].quantile(0.99)
            print(f"- {col}: skew={skew:.3f}, 99th_pct={pct_99}")

    # Categorical summaries
    cat_cols = [c for c in features if c in df.select_dtypes(include=[object, "category"]).columns]
    if cat_cols:
        print("\nCategorical feature top levels:")
        for col in cat_cols:
            vc = df[col].value_counts(dropna=False).head(10)
            print(f"- {col}: {vc.to_dict()}")


if __name__ == "__main__":
    inspect_features()
