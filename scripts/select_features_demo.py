"""Demonstration script for selecting numerical and categorical features.

Run: python scripts/select_features_demo.py
"""
import pandas as pd


def build_sample():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "tenure": [12, 3, 45, 24],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": [29.85, 188.95, 108.15, 1840.75],
            "gender": ["Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "Churn": ["No", "Yes", "No", "No"],
        }
    )
    return df


def classify_features(df, target_column="Churn"):
    excluded = ["CustomerID"]
    candidate = [c for c in df.columns if c not in excluded and c != target_column]

    numerical = []
    categorical = []

    for c in candidate:
        col = df[c]
        # heuristic: low unique values and object or integer -> categorical
        if pd.api.types.is_object_dtype(col) or (pd.api.types.is_integer_dtype(col) and col.nunique() < 10):
            categorical.append(c)
        else:
            numerical.append(c)

    return {
        "excluded": excluded,
        "numerical": numerical,
        "categorical": categorical,
        "target": target_column,
    }


def main():
    df = build_sample()
    print("Sample dataframe:\n", df)
    result = classify_features(df)
    print("\nDetected feature groups:")
    for k, v in result.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
