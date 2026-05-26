import sys
from typing import List
import numpy as np
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(
    train_file: str = typer.Argument(..., help="Training set CSV."),
    val_file: str = typer.Argument(..., help="Validation set CSV."),
    layers: List[int] = typer.Option([24, 24], "--layer", "-l", help="Hidden layer sizes."),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs."),
    loss: str = typer.Option("categoricalCrossentropy", "--loss", help="Loss function."),
    learning_rate: float = typer.Option(0.01, "--learning-rate", help="Learning rate."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Mini-batch size."),
    weights_initializer: str = typer.Option("heUniform", "--weights-initializer", help="Weight init method."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    # TODO: load data, preprocess, build network, train, print metrics, save model
    pass


if __name__ == "__main__":
    app()
