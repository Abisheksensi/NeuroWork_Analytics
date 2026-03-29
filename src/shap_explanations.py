from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def generate_shap_summary():
    model = joblib.load("models/random_forest.pkl")
    X_test_sample = pd.read_csv("data/processed/X_test.csv").head(100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values[1], X_test_sample, show=False)
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()


def explain_single_prediction(input_df):
    model = joblib.load("models/random_forest.pkl")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    single_row_values = shap_values[1][0]
    explanations = []

    for feature, shap_value in zip(input_df.columns, single_row_values):
        explanations.append(
            {
                "feature": feature,
                "shap_value": float(shap_value),
                "direction": (
                    "increases risk" if shap_value > 0 else "decreases risk"
                ),
            }
        )

    explanations.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return explanations[:5]


if __name__ == "__main__":
    generate_shap_summary()
