from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Path to the dataset CSV."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
    features: List[int] = typer.Option([], "--feature", "-f", help="Feature column indices to plot (default: first 5)."),
):
    df = pd.read_csv(dataset_file, header=None)
    df = df.drop(columns=[0])  # drop sample ID

    if not features:
        all_features = [c for c in df.columns if c != label_col]
        features = all_features[:5]

    subset = df[list(features) + [label_col]]
    sns.pairplot(subset, hue=label_col)
    plt.show()


if __name__ == "__main__":
    app()
