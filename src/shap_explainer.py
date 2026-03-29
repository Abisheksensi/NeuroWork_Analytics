import joblib
import numpy as np
import shap


def explain_single_prediction(input_df):
    model = joblib.load("models/random_forest.pkl")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    print("SHAP values shape:", type(shap_values))

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_row = shap_values[0]
    else:
        raise ValueError(f"Unsupported SHAP output format: {type(shap_values)}")

    explanations = []

    for feature, shap_value in zip(input_df.columns, shap_row):
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
