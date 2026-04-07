"""Assignment-grade scaling demo: full workflow using StandardScaler and ColumnTransformer.

Produces saved scaler and shows correct vs incorrect scaling scores.
Run: python scripts/scale_demo.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os


def make_data(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    tenure = rng.randint(0, 72, size=n)
    monthly = rng.normal(70, 30, size=n).clip(0)
    total = (tenure * monthly) + rng.normal(0, 50, size=n)
    gender = rng.choice(["Male", "Female"], size=n)
    churn = (0.02 * tenure + 0.01 * monthly + rng.normal(0, 1, size=n) < 1.5).astype(int)
    df = pd.DataFrame({"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total, "gender": gender, "Churn": churn})
    return df


def incorrect_scaling(df):
    X = df.drop(columns=["Churn"]) 
    y = df["Churn"]
    # WRONG: scale before split -> leakage
    scaler = StandardScaler()
    X_num = X[["tenure", "MonthlyCharges", "TotalCharges"]].fillna(0)
    X_num_scaled = scaler.fit_transform(X_num)
    Xp = np.hstack([X_num_scaled, pd.get_dummies(X["gender"]).values])
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.3, random_state=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def correct_pipeline(df, save_dir="models"):
    X = df.drop(columns=["Churn"]) 
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(save_dir, "standard_preprocessor.pkl"))
    joblib.dump(pipeline.named_steps["clf"], os.path.join(save_dir, "logistic_model.pkl"))

    return score


def main():
    print("Generating synthetic churn dataset...")
    df = make_data()

    print("Running incorrect (leaky) scaling workflow...")
    bad_score = incorrect_scaling(df)

    print("Running correct pipeline with ColumnTransformer and StandardScaler...")
    good_score = correct_pipeline(df)

    print(f"Leaky workflow accuracy: {bad_score:.3f}")
    print(f"Correct pipeline accuracy: {good_score:.3f}")
    print("Saved preprocessor and model into models/ directory.")


if __name__ == "__main__":
    main()
