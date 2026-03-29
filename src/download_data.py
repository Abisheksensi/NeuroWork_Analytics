from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_NAME = "osmi/mental-health-in-tech-survey"
DOWNLOAD_DIR = Path("data/raw")
DATA_FILE = DOWNLOAD_DIR / "survey.csv"


def ensure_dataset_downloaded() -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        return DATA_FILE

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET_NAME,
        path=str(DOWNLOAD_DIR),
        unzip=True,
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset download completed, but {DATA_FILE} was not found."
        )

    return DATA_FILE


if __name__ == "__main__":
    dataset_path = ensure_dataset_downloaded()
    print(f"Dataset ready at {dataset_path}")
