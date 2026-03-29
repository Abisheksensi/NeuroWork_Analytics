from pathlib import Path

import joblib
import pandas as pd
from clean_data import clean_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def handle_missing_values(df):
    imputed_df = df.copy()

    missing_before = imputed_df.isna().sum()
    missing_before = missing_before[missing_before > 0]
    print("Missing values before imputation:")
    print(missing_before)

    for column in imputed_df.columns:
        if imputed_df[column].dtype == "object":
            imputed_df[column] = imputed_df[column].fillna(imputed_df[column].mode()[0])
        elif imputed_df[column].dtype in ["int64", "float64"]:
            imputed_df[column] = imputed_df[column].fillna(imputed_df[column].median())

    missing_after = imputed_df.isna().sum()
    missing_after = missing_after[missing_after > 0]
    print("Missing values after imputation:")
    print(missing_after)

    return imputed_df


def encode_features(df):
    encoded_df = df.copy()

    binary_columns = [
        "self_employed",
        "family_history",
        "remote_work",
        "tech_company",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "mental_health_consequence",
        "phys_health_consequence",
        "mental_vs_physical",
        "obs_consequence",
    ]

    for column in binary_columns:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column])

    encoded_df["work_interfere"] = encoded_df["work_interfere"].map(
        {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    )
    encoded_df["leave"] = encoded_df["leave"].map(
        {
            "Very easy": 0,
            "Somewhat easy": 1,
            "Don't know": 2,
            "Somewhat difficult": 3,
            "Very difficult": 4,
        }
    )
    encoded_df["no_employees"] = encoded_df["no_employees"].map(
        {
            "1-5": 0,
            "6-25": 1,
            "26-100": 2,
            "100-500": 3,
            "500-1000": 4,
            "More than 1000": 5,
        }
    )

    gender_dummies = pd.get_dummies(
        encoded_df["Gender"],
        prefix="Gender",
        drop_first=False,
    )
    encoded_df = encoded_df.drop(columns=["Gender"])
    encoded_df = pd.concat([encoded_df, gender_dummies], axis=1)

    additional_label_columns = [
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
    ]

    for column in additional_label_columns:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column])

    feature_columns = [column for column in encoded_df.columns if column != "treatment"]
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_columns, models_dir / "feature_columns.pkl")

    return encoded_df


def split_data(df):
    split_df = df.copy()
    X = split_df.drop(columns=["treatment"])

    encoder = LabelEncoder()
    y = pd.Series(
        encoder.fit_transform(split_df["treatment"]),
        name="treatment",
        index=split_df.index,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame().to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_frame().to_csv(output_dir / "y_test.csv", index=False)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def run_preprocessing():
    dataframe = pd.read_csv("data/raw/survey.csv")
    cleaned_df = clean_data(dataframe)
    imputed_df = handle_missing_values(cleaned_df)
    encoded_df = encode_features(imputed_df)
    X_train, X_test, y_train, y_test = split_data(encoded_df)

    print("Preprocessing complete.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
