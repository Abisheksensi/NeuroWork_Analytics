from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shap_explainer import explain_single_prediction


rf_model = joblib.load(PROJECT_ROOT / "models/random_forest.pkl")
lr_model = joblib.load(PROJECT_ROOT / "models/logistic_regression.pkl")
scaler = joblib.load(PROJECT_ROOT / "models/scaler.pkl")
feature_columns = joblib.load(PROJECT_ROOT / "models/feature_columns.pkl")

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Mental Health Prediction API is running",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            raise ValueError("Request JSON body is empty.")

        input_df = pd.DataFrame([input_json]).reindex(
            columns=feature_columns,
            fill_value=0,
        )

        rf_prediction = rf_model.predict(input_df)[0]
        rf_probability = float(rf_model.predict_proba(input_df)[0][1])

        scaled_input = scaler.transform(input_df)
        lr_prediction = lr_model.predict(scaled_input)[0]
        lr_probability = float(lr_model.predict_proba(scaled_input)[0][1])

        shap_explanation = explain_single_prediction(input_df)

        return jsonify(
            {
                "prediction": (
                    "Likely to seek treatment"
                    if rf_prediction == 1
                    else "Unlikely to seek treatment"
                ),
                "probability": rf_probability,
                "confidence_percent": int(round(rf_probability * 100)),
                "model_used": "Random Forest",
                "shap_explanation": shap_explanation,
                "lr_prediction": (
                    "Likely to seek treatment"
                    if lr_prediction == 1
                    else "Unlikely to seek treatment"
                ),
                "lr_probability": lr_probability,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=False)
