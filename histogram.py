import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Path to the dataset CSV."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    df = pd.read_csv(dataset_file, header=None)
    df = df.drop(columns=[0])  # drop sample ID

    features = [c for c in df.columns if c != label_col]
    n = len(features)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.histplot(data=df, x=feat, hue=label_col, ax=axes[i], kde=True, bins=20)
        axes[i].set_title(f"feature {feat}")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature distributions by class")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()
