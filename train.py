import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/live.csv"
MODEL_PATH = "models/reaction_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    # Keep only the required columns
    df = df[
        ["status_type", "num_comments", "num_shares", "num_reactions"]
    ].dropna().drop_duplicates()

    X = pd.get_dummies(
        df[["status_type", "num_comments", "num_shares"]],
        columns=["status_type"],
        dtype=int
    )
    y = df["num_reactions"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "columns": X.columns.tolist()},
        MODEL_PATH
    )

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
