from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    training_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {training_accuracy}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "random_forest.pkl")
    print("Random Forest model saved.")


if __name__ == "__main__":
    train_random_forest()
