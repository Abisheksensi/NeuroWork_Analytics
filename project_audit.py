import os
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "project_audit.txt"
OUTPUT_LINES = []
ISSUES = []
STATE = {
    "raw_shape": None,
    "raw_columns": None,
    "target_distribution": None,
    "processed_columns": None,
    "processed_shapes": {},
    "models": {},
    "feature_columns": None,
    "top_features": None,
    "frontend_files": [],
}


def log(message=""):
    print(message)
    OUTPUT_LINES.append(str(message))


def format_exception(step_name, exc):
    message = f"{step_name} failed: {exc}"
    ISSUES.append(message)
    log(message)


def print_header(title):
    log(f"\n{'=' * 20} {title} {'=' * 20}")


def step_1_read_folder_structure():
    step_name = "Step 1"
    print_header("STEP 1 — FOLDER STRUCTURE")
    try:
        log(str(PROJECT_ROOT))
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs.sort()
            files.sort()
            root_path = Path(root)
            level = len(root_path.relative_to(PROJECT_ROOT).parts)
            indent = "    " * level
            if root_path != PROJECT_ROOT:
                log(f"{indent}{root_path.name}/")
            for file_name in files:
                log(f"{indent}    {file_name}")
    except Exception as exc:
        format_exception(step_name, exc)


def step_2_read_raw_data():
    step_name = "Step 2"
    print_header("STEP 2 — RAW DATA")
    try:
        raw_path = PROJECT_ROOT / "data/raw/survey.csv"
        raw_df = pd.read_csv(raw_path)

        STATE["raw_shape"] = raw_df.shape
        STATE["raw_columns"] = list(raw_df.columns)
        STATE["target_distribution"] = raw_df["treatment"].value_counts(dropna=False)

        log(f"Shape: {raw_df.shape}")
        log("Column names:")
        for column in raw_df.columns:
            log(f"- {column}")

        log("Data types:")
        log(raw_df.dtypes.to_string())

        log("Missing value count per column:")
        log(raw_df.isna().sum().to_string())

        log("Treatment value counts:")
        log(raw_df["treatment"].value_counts(dropna=False).to_string())

        log("First 3 rows:")
        log(raw_df.head(3).to_string())
    except Exception as exc:
        format_exception(step_name, exc)


def step_3_read_processed_data():
    step_name = "Step 3"
    print_header("STEP 3 — PROCESSED DATA")
    try:
        processed_dir = PROJECT_ROOT / "data/processed"
        X_train = pd.read_csv(processed_dir / "X_train.csv")
        X_test = pd.read_csv(processed_dir / "X_test.csv")
        y_train = pd.read_csv(processed_dir / "y_train.csv")
        y_test = pd.read_csv(processed_dir / "y_test.csv")

        STATE["processed_shapes"] = {
            "X_train": X_train.shape,
            "X_test": X_test.shape,
            "y_train": y_train.shape,
            "y_test": y_test.shape,
        }
        STATE["processed_columns"] = list(X_train.columns)

        log(f"X_train shape: {X_train.shape}")
        log(f"X_test shape: {X_test.shape}")
        log(f"y_train shape: {y_train.shape}")
        log(f"y_test shape: {y_test.shape}")

        log("X_train columns:")
        for column in X_train.columns:
            log(f"- {column}")

        log("y_train value counts:")
        log(y_train.iloc[:, 0].value_counts(dropna=False).to_string())

        log("y_test value counts:")
        log(y_test.iloc[:, 0].value_counts(dropna=False).to_string())

        log("First 2 rows of X_train:")
        log(X_train.head(2).to_string())
    except Exception as exc:
        format_exception(step_name, exc)


def step_4_read_saved_models():
    step_name = "Step 4"
    print_header("STEP 4 — SAVED MODELS")
    try:
        rf_model = joblib.load(PROJECT_ROOT / "models/random_forest.pkl")
        lr_model = joblib.load(PROJECT_ROOT / "models/logistic_regression.pkl")
        scaler = joblib.load(PROJECT_ROOT / "models/scaler.pkl")
        feature_columns = joblib.load(PROJECT_ROOT / "models/feature_columns.pkl")

        STATE["models"] = {
            "random_forest": rf_model,
            "logistic_regression": lr_model,
            "scaler": scaler,
        }
        STATE["feature_columns"] = list(feature_columns)

        log(f"Random Forest model type: {type(rf_model)}")
        log(f"Random Forest parameters: {rf_model.get_params()}")
        log(f"Logistic Regression model type: {type(lr_model)}")
        log(f"Logistic Regression parameters: {lr_model.get_params()}")

        scaler_features = getattr(scaler, "n_features_in_", "unknown")
        log(f"Scaler expected feature count: {scaler_features}")

        log("Feature columns in order:")
        for column in feature_columns:
            log(f"- {column}")
    except Exception as exc:
        format_exception(step_name, exc)


def print_file_contents(file_path):
    header = f"===== FILE: {file_path.relative_to(PROJECT_ROOT)} ====="
    log(header)
    try:
        log(file_path.read_text())
    except Exception as exc:
        message = f"Could not read {file_path}: {exc}"
        ISSUES.append(message)
        log(message)


def step_5_read_python_source_files():
    step_name = "Step 5"
    print_header("STEP 5 — PYTHON SOURCE FILES")
    try:
        source_dirs = [PROJECT_ROOT / "src", PROJECT_ROOT / "app/backend"]
        for source_dir in source_dirs:
            if not source_dir.exists():
                message = f"Missing source directory: {source_dir}"
                ISSUES.append(message)
                log(message)
                continue

            for file_path in sorted(source_dir.rglob("*.py")):
                print_file_contents(file_path)
    except Exception as exc:
        format_exception(step_name, exc)


def step_6_read_frontend_files():
    step_name = "Step 6"
    print_header("STEP 6 — FRONTEND FILES")
    try:
        frontend_files = [
            PROJECT_ROOT / "app/frontend/index.html",
            PROJECT_ROOT / "app/frontend/style.css",
            PROJECT_ROOT / "app/frontend/script.js",
        ]
        STATE["frontend_files"] = [str(path.relative_to(PROJECT_ROOT)) for path in frontend_files]
        for file_path in frontend_files:
            print_file_contents(file_path)
    except Exception as exc:
        format_exception(step_name, exc)


def step_7_run_live_prediction_test():
    step_name = "Step 7"
    print_header("STEP 7 — LIVE PREDICTION TEST")
    try:
        rf_model = STATE["models"].get("random_forest")
        feature_columns = STATE["feature_columns"]

        if rf_model is None or feature_columns is None:
            raise ValueError("Random Forest model or feature columns are not loaded.")

        sample_input = {column: 0 for column in feature_columns}
        sample_input["Age"] = 28
        sample_input["family_history"] = 1
        sample_input["work_interfere"] = 2
        sample_input["benefits"] = 1
        sample_input["Gender_Male"] = 1
        sample_input["Gender_Female"] = 0
        sample_input["Gender_Other"] = 0

        input_df = pd.DataFrame([sample_input], columns=feature_columns)
        prediction = rf_model.predict(input_df)[0]
        probability = rf_model.predict_proba(input_df)[0][1]

        log(f"Predicted class: {prediction}")
        log(f"Prediction probability: {probability}")

        feature_importances = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": rf_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        top_features = feature_importances.head(5)
        STATE["top_features"] = top_features

        log("Top 5 feature importances:")
        log(top_features.to_string(index=False))
    except Exception as exc:
        format_exception(step_name, exc)


def step_8_check_outputs():
    step_name = "Step 8"
    print_header("STEP 8 — OUTPUT FILES")
    try:
        output_dir = PROJECT_ROOT / "outputs"
        plots_dir = output_dir / "plots"

        log("Files in outputs/:")
        if output_dir.exists():
            for path in sorted(output_dir.iterdir()):
                if path.is_file():
                    log(f"- {path.name}")
        else:
            message = "outputs/ directory is missing."
            ISSUES.append(message)
            log(message)

        log("Files in outputs/plots/:")
        if plots_dir.exists():
            for path in sorted(plots_dir.iterdir()):
                if path.is_file():
                    size_kb = path.stat().st_size / 1024
                    log(f"- {path.name}: {size_kb:.2f} KB")
        else:
            message = "outputs/plots/ directory is missing."
            ISSUES.append(message)
            log(message)
    except Exception as exc:
        format_exception(step_name, exc)


def step_9_print_summary_report():
    step_name = "Step 9"
    print_header("STEP 9 — SUMMARY REPORT")
    try:
        log("Dataset:")
        if STATE["raw_shape"] and STATE["raw_columns"] is not None:
            log(f"- Rows, columns: {STATE['raw_shape']}")
            log(f"- Target distribution:\n{STATE['target_distribution'].to_string()}")
        else:
            log("- Raw dataset summary unavailable.")

        log("Preprocessing:")
        if STATE["processed_columns"] is not None:
            log(f"- Columns after preprocessing ({len(STATE['processed_columns'])}):")
            for column in STATE["processed_columns"]:
                log(f"  - {column}")
            log("- Encodings applied: label encoding, ordinal encoding, and one-hot encoding for Gender.")
        else:
            log("- Processed data summary unavailable.")

        log("Models:")
        rf_model = STATE["models"].get("random_forest")
        lr_model = STATE["models"].get("logistic_regression")
        if rf_model is not None:
            log(
                "- Random Forest: "
                f"{type(rf_model).__name__}, "
                f"n_estimators={rf_model.get_params().get('n_estimators')}, "
                f"class_weight={rf_model.get_params().get('class_weight')}, "
                f"random_state={rf_model.get_params().get('random_state')}"
            )
        else:
            log("- Random Forest summary unavailable.")
        if lr_model is not None:
            log(
                "- Logistic Regression: "
                f"{type(lr_model).__name__}, "
                f"max_iter={lr_model.get_params().get('max_iter')}, "
                f"class_weight={lr_model.get_params().get('class_weight')}, "
                f"random_state={lr_model.get_params().get('random_state')}"
            )
            if hasattr(lr_model, "n_iter_"):
                log(f"- Logistic Regression fitted iterations: {lr_model.n_iter_}")
        else:
            log("- Logistic Regression summary unavailable.")
        log("- Training accuracy if available: not stored separately in saved artifacts.")

        log("Features:")
        if STATE["feature_columns"] is not None:
            log(f"- Total feature count: {len(STATE['feature_columns'])}")
        else:
            log("- Feature count unavailable.")
        if STATE["top_features"] is not None:
            log("- Top 5 by importance:")
            log(STATE["top_features"].to_string(index=False))
        else:
            log("- Top feature importances unavailable.")

        log("Frontend:")
        if STATE["frontend_files"]:
            for file_name in STATE["frontend_files"]:
                log(f"- {file_name}")
        else:
            log("- No frontend files found.")

        log("Issues found:")
        if ISSUES:
            for issue in ISSUES:
                log(f"- {issue}")
        else:
            log("- No issues found.")
    except Exception as exc:
        format_exception(step_name, exc)


def save_report():
    REPORT_PATH.write_text("\n".join(OUTPUT_LINES))
    log(f"\nReport saved to: {REPORT_PATH}")


def main():
    step_1_read_folder_structure()
    step_2_read_raw_data()
    step_3_read_processed_data()
    step_4_read_saved_models()
    step_5_read_python_source_files()
    step_6_read_frontend_files()
    step_7_run_live_prediction_test()
    step_8_check_outputs()
    step_9_print_summary_report()
    REPORT_PATH.write_text("\n".join(OUTPUT_LINES))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
