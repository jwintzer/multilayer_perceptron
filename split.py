import sys
import numpy as np
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Path to the raw dataset CSV."),
    train_ratio: float = typer.Option(0.8, "--ratio", "-r", help="Fraction of data used for training."),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    # TODO: load CSV, shuffle, split into train/val, save both to disk
    pass


if __name__ == "__main__":
    app()
