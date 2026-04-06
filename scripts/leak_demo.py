"""Simple demo showing data leakage vs correct pipeline.

Run: python scripts/leak_demo.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_dataset(n=1000, leak=False):
    rng = np.random.RandomState(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # target depends on x1
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n) > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    if leak:
        # add a target-derived feature: exists only when y==1
        df["days_until_event"] = np.where(df.y == 1, rng.randint(1, 30, size=n), np.nan)
    else:
        df["days_until_event"] = np.nan

    return df


def eval_leak_pipeline(df):
    # Incorrect: fill na with 0 then scale before split (train-test contamination)
    X = df[["x1", "x2", "days_until_event"]].fillna(0)
    y = df.y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit on full data -> contamination
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score


def eval_correct_pipeline(df):
    # Correct: split first, then fit scaler on train; drop target-derived feature
    X = df[["x1", "x2", "days_until_event"]]
    y = df.y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # drop the clearly leaking feature (prediction-moment test)
    X_train = X_train[["x1", "x2"]]
    X_test = X_test[["x1", "x2"]]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    return score


def main():
    print("Creating dataset with target-derived leak feature...")
    df = create_dataset(n=2000, leak=True)

    leak_score = eval_leak_pipeline(df)
    correct_score = eval_correct_pipeline(df)

    print(f"Leaky pipeline accuracy: {leak_score:.3f}")
    print(f"Correct pipeline accuracy: {correct_score:.3f}")


if __name__ == "__main__":
    main()
