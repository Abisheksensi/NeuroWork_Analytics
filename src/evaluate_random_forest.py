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


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, average="weighted"),
        "Recall": recall_score(y_true, predictions, average="weighted"),
        "F1": f1_score(y_true, predictions, average="weighted"),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
    }


def evaluate_rf():
    rf_model = joblib.load("models/random_forest.pkl")
    lr_model = joblib.load("models/logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")

    rf_predictions = rf_model.predict(X_test)
    rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = calculate_metrics(y_test, rf_predictions, rf_probabilities)

    print(f"Accuracy: {rf_metrics['Accuracy']}")
    print(f"Precision (weighted): {rf_metrics['Precision']}")
    print(f"Recall (weighted): {rf_metrics['Recall']}")
    print(f"F1 Score (weighted): {rf_metrics['F1']}")
    print(f"ROC-AUC Score: {rf_metrics['ROC_AUC']}")

    output_dir = Path("outputs")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    confusion = confusion_matrix(y_test, rf_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(plots_dir / "rf_confusion_matrix.png")
    plt.close()

    X_test_scaled = scaler.transform(X_test)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = calculate_metrics(y_test, lr_predictions, lr_probabilities)

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Accuracy": lr_metrics["Accuracy"],
                "Precision": lr_metrics["Precision"],
                "Recall": lr_metrics["Recall"],
                "F1": lr_metrics["F1"],
                "ROC_AUC": lr_metrics["ROC_AUC"],
            },
            {
                "Model": "Random Forest",
                "Accuracy": rf_metrics["Accuracy"],
                "Precision": rf_metrics["Precision"],
                "Recall": rf_metrics["Recall"],
                "F1": rf_metrics["F1"],
                "ROC_AUC": rf_metrics["ROC_AUC"],
            },
        ],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
    )
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)


if __name__ == "__main__":
    evaluate_rf()
