from pathlib import Path

import pandas as pd

from download_data import ensure_dataset_downloaded


def load_raw_data(filepath):
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please download it from Kaggle."
        )

    dataframe = pd.read_csv(path)

    print(f"Dataset shape: {dataframe.shape}")
    print(f"Column names: {list(dataframe.columns)}")
    print("First 5 rows:")
    print(dataframe.head())
    print("Data types:")
    print(dataframe.dtypes)

    return dataframe


if __name__ == "__main__":
    dataset_path = ensure_dataset_downloaded()
    load_raw_data(dataset_path)
