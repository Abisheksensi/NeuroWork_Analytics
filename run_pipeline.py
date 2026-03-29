from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    data_path = PROJECT_ROOT / "data/raw/survey.csv"
    processed_path = PROJECT_ROOT / "data/processed/X_train.csv"
    lr_model_path = PROJECT_ROOT / "models/logistic_regression.pkl"
    rf_model_path = PROJECT_ROOT / "models/random_forest.pkl"

    print("[STEP 1] Running: checking raw dataset...")
    if not data_path.exists():
        try:
            from download_data import ensure_dataset_downloaded

            ensure_dataset_downloaded()
        except Exception as exc:
            print(f"Error: unable to prepare data/raw/survey.csv. {exc}")
            raise SystemExit(1)
    print("[STEP 1] Done.")

    print("[STEP 2] Running: preprocessing data...")
    if not processed_path.exists():
        from preprocess import run_preprocessing

        run_preprocessing()
    print("[STEP 2] Done.")

    print("[STEP 3] Running: training logistic regression...")
    if not lr_model_path.exists():
        from train_logistic_regression import train_logistic_regression

        train_logistic_regression()
    print("[STEP 3] Done.")

    print("[STEP 4] Running: training random forest...")
    if not rf_model_path.exists():
        from train_random_forest import train_random_forest

        train_random_forest()
    print("[STEP 4] Done.")

    print("[STEP 5] Running: evaluating logistic regression...")
    from evaluate_logistic_regression import evaluate_lr

    evaluate_lr()
    print("[STEP 5] Done.")

    print("[STEP 6] Running: evaluating random forest...")
    from evaluate_random_forest import evaluate_rf

    evaluate_rf()
    print("[STEP 6] Done.")

    print("[STEP 7] Running: plotting feature importance...")
    from feature_importance import plot_feature_importance

    plot_feature_importance()
    print("[STEP 7] Done.")

    print("Pipeline complete. Start the API with: python app/backend/app.py")


if __name__ == "__main__":
    main()
