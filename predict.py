import typer

import model
import preprocessing as prep

app = typer.Typer()


@app.command()
def main(
    dataset_file: str = typer.Argument(..., help="Dataset to run predictions on."),
    model_file: str = typer.Option("saved_model.npy", "--model", "-m", help="Path to saved model file."),
    label_col: int = typer.Option(1, "--label-col", help="Index of the label column."),
):
    # load network weights + preprocessing params that were saved with it
    network, mean, std, classes = model.load_model(model_file)

    # load data and apply the exact same normalization used during training
    X_raw, y_raw = prep.load_data(dataset_file, label_col)
    X, _, _      = prep.normalize(X_raw, mean, std)

    # inference
    y_pred = model.predict(network, X)

    # binary ground-truth labels: B -> 0, M -> 1 (matches class index order)
    y_true = prep.to_binary_labels(y_raw, list(classes))

    # P(malignant) = softmax output at the index of "M" in the classes list
    m_idx      = list(classes).index("M")
    p_malignant = y_pred[:, m_idx]

    # evaluate — binary cross-entropy as specified in the subject
    bce      = model.binary_cross_entropy(p_malignant, y_true)
    pred_cls = y_pred.argmax(axis=1)
    accuracy = (pred_cls == y_true).mean()

    print(f"binary cross-entropy : {bce:.4f}")
    print(f"accuracy             : {accuracy:.4f}")


if __name__ == "__main__":
    app()
