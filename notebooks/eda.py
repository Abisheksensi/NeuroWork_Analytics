from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from download_data import ensure_dataset_downloaded


OUTPUT_DIR = PROJECT_ROOT / "outputs/plots"


def main() -> None:
    data_path = ensure_dataset_downloaded()
    dataframe = pd.read_csv(data_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.countplot(data=dataframe, x="treatment")
    plt.title("Treatment Distribution")
    plt.xlabel("Treatment")
    plt.ylabel("Count")
    plt.savefig(OUTPUT_DIR / "treatment_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(dataframe["Age"].dropna(), bins=30, edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig(OUTPUT_DIR / "age_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    top_genders = dataframe["Gender"].value_counts().head(10).index
    gender_data = dataframe[dataframe["Gender"].isin(top_genders)]
    sns.countplot(data=gender_data, y="Gender", order=top_genders)
    plt.title("Gender Distribution (Top 10 Values)")
    plt.xlabel("Count")
    plt.ylabel("Gender")
    plt.savefig(OUTPUT_DIR / "gender_distribution.png")
    plt.close()

    missing_counts = dataframe.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values()

    plt.figure(figsize=(10, 6))
    plt.barh(missing_counts.index, missing_counts.values)
    plt.title("Missing Value Counts by Column")
    plt.xlabel("Missing Value Count")
    plt.ylabel("Column")
    plt.savefig(OUTPUT_DIR / "missing_values.png")
    plt.close()


if __name__ == "__main__":
    main()
