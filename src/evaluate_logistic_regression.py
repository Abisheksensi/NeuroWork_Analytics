from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_lr():
    model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision (weighted): {precision_score(y_test, predictions, average='weighted')}")
    print(f"Recall (weighted): {recall_score(y_test, predictions, average='weighted')}")
    print(f"F1 Score (weighted): {f1_score(y_test, predictions, average='weighted')}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probabilities)}")

    confusion = confusion_matrix(y_test, predictions)
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title("Logistic Regression — Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(output_dir / "lr_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    evaluate_lr()
