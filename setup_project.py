from pathlib import Path


DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "outputs/plots",
    "app/backend",
    "app/frontend",
]


def create_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")
        return

    path.mkdir(parents=True, exist_ok=True)


def create_gitkeep(path: Path) -> None:
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
        print(f"Created: {gitkeep}")


def main() -> None:
    for directory in DIRECTORIES:
        path = Path(directory)
        create_directory(path)
        create_gitkeep(path)


if __name__ == "__main__":
    main()
