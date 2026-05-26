import sys
import numpy as np
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Dataset to run predictions on."),
    model_dir: str = typer.Option("saved_model", "--model", "-m", help="Path to saved model directory."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    # TODO: load model, preprocess, predict, evaluate with binary cross-entropy
    pass


if __name__ == "__main__":
    app()
