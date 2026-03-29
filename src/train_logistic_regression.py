from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_logistic_regression():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    training_accuracy = model.score(X_train_scaled, y_train)
    print(f"Training accuracy: {training_accuracy}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "logistic_regression.pkl")
    print("Logistic Regression model saved.")

    joblib.dump(scaler, models_dir / "scaler.pkl")
    print("Scaler saved.")


if __name__ == "__main__":
    train_logistic_regression()
