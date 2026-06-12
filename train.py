from typing import List
import typer

import layers as lyr
import model
import preprocessing as prep

app = typer.Typer()


@app.command()
def main(
    train_file: str = typer.Argument(..., help="Training set CSV."),
    val_file: str = typer.Argument(..., help="Validation set CSV."),
    hidden_layers: List[int] = typer.Option([24, 24], "--layer", "-l", help="Hidden layer sizes."),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs."),
    loss: str = typer.Option("categoricalCrossentropy", "--loss", help="Loss function."),
    learning_rate: float = typer.Option(0.01, "--learning-rate", help="Learning rate."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Mini-batch size."),
    weights_initializer: str = typer.Option("heUniform", "--weights-initializer", help="Weight init method."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    # --- data ---
    X_train_raw, y_train_raw = prep.load_data(train_file, label_col)
    X_val_raw,   y_val_raw   = prep.load_data(val_file,   label_col)

    # fit normalization on train only, then apply same transform to val
    X_train, mean, std = prep.normalize(X_train_raw)
    X_val,   _,   _   = prep.normalize(X_val_raw, mean, std)

    # one-hot encode — class ordering is derived from train and applies to both
    y_train, classes = prep.encode_labels(y_train_raw)
    y_val,   _       = prep.encode_labels(y_val_raw)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")

    # --- network ---
    n_features = X_train.shape[1]
    n_classes  = len(classes)

    layer_list = [lyr.DenseLayer(n_features, activation="sigmoid")]
    for size in hidden_layers:
        layer_list.append(
            lyr.DenseLayer(size, activation="sigmoid", weights_initializer=weights_initializer)
        )
    layer_list.append(
        lyr.DenseLayer(n_classes, activation="softmax", weights_initializer=weights_initializer)
    )

    network = model.createNetwork(layer_list)

    # --- train ---
    model.fit(
        network, X_train, y_train, X_val, y_val,
        loss=loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

    # --- save ---
    model.save_model(network, mean, std, classes)


if __name__ == "__main__":
    app()
