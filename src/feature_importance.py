from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_feature_importance():
    model = joblib.load("models/random_forest.pkl")
    X_train = pd.read_csv("data/processed/X_train.csv")

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    output_dir = Path("outputs")
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    top_features = importance_df.head(15).sort_values(by="importance", ascending=True)
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        palette="Blues_r",
        hue="feature",
        dodge=False,
        legend=False,
    )
    plt.title("Top 15 Feature Importances — Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.savefig(plots_dir / "feature_importance.png")
    plt.close()

    print("Top 5 features and their scores:")
    print(importance_df.head(5).to_string(index=False))


if __name__ == "__main__":
    plot_feature_importance()
