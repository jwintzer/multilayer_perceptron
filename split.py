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
):
    df = pd.read_csv(dataset_file, header=None)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]

    train_df.to_csv("train.csv", index=False, header=False)
    val_df.to_csv("val.csv", index=False, header=False)

    print(f"split: {len(train_df)} train / {len(val_df)} val")


if __name__ == "__main__":
    app()
