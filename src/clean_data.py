from pathlib import Path

import pandas as pd


def standardize_gender(value):
    text = str(value).strip().lower()

    if "female" in text or text == "f":
        return "Female"

    if "male" in text or text == "m":
        return "Male"

    return "Other"


def clean_data(df):
    print(f"Shape before cleaning: {df.shape}")

    cleaned_df = df.drop(columns=["Timestamp", "comments", "state", "Country"])
    cleaned_df = cleaned_df[cleaned_df["Age"].between(18, 75, inclusive="both")]
    cleaned_df["Gender"] = cleaned_df["Gender"].apply(standardize_gender)

    print(f"Shape after cleaning: {cleaned_df.shape}")

    return cleaned_df


if __name__ == "__main__":
    data_path = Path("data/raw/survey.csv")
    dataframe = pd.read_csv(data_path)
    cleaned_dataframe = clean_data(dataframe)
    print(cleaned_dataframe.head())
