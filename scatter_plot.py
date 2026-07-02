import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Path to the dataset CSV."),
    x: int = typer.Option(2, "--x", help="Column index for the x-axis feature."),
    y: int = typer.Option(3, "--y", help="Column index for the y-axis feature."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    df = pd.read_csv(dataset_file, header=None)
    df = df.drop(columns=[0])  # drop sample ID

    sns.scatterplot(data=df, x=x, y=y, hue=label_col)
    plt.title(f"feature {x} vs feature {y}")
    plt.show()


if __name__ == "__main__":
    app()
