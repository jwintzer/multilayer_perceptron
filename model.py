import numpy as np
import matplotlib.pyplot as plt
from layers import DenseLayer


def createNetwork(layer_list):
    """
    Initialize a network from a list of DenseLayer objects.
    The first layer is the input layer (defines feature count, no weights).
    Every subsequent layer gets its weights initialized against the previous layer.
    Returns the same list — the network is just a list of layers.
    """
    for i in range(1, len(layer_list)):
        n_inputs = layer_list[i - 1].n_neurons
        layer_list[i].init_weights(n_inputs)
    return layer_list


def forward(network, X):
    """
    Full forward pass: run X through every layer in sequence.
    Returns the output of the last layer (softmax probabilities).
    """
    out = X
    for layer in network:
        out = layer.forward(out)
    return out


def categorical_cross_entropy(y_pred, y_true):
    """
    Categorical cross-entropy between softmax output and one-hot labels.
    y_pred : (batch, n_classes)
    y_true : (batch, n_classes)
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def binary_cross_entropy(y_pred, y_true):
    """
    Binary cross-entropy for evaluation in predict.py.
    E = -1/N * sum[ y*log(p) + (1-y)*log(1-p) ]
    y_pred : (n,)  — P(positive class), i.e. softmax output[:, 1]
    y_true : (n,)  — 0 or 1
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_accuracy(y_pred, y_true):
    """
    Classification accuracy.
    y_pred : (batch, n_classes) — softmax output
    y_true : (batch, n_classes) — one-hot labels
    """
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(pred_classes == true_classes)


def predict(network, X):
    """Run inference — forward pass with no gradient tracking."""
    return forward(network, X)


def _backward(network, y_pred, y_true, learning_rate):
    """
    Backpropagation through the whole network.
    Updates weights and biases of every layer in place.

    For the output layer (softmax + categorical cross-entropy) the combined
    gradient simplifies cleanly to:  delta = y_pred - y_true

    For every hidden layer below, the chain rule gives:
        delta_prev = (W.T @ delta.T).T * activation'(z_prev)
    """
    batch_size = y_pred.shape[0]

    # output layer: softmax + categorical CE combined gradient
    delta = y_pred - y_true  # (batch, n_out)

    for i in range(len(network) - 1, 0, -1):
        layer = network[i]

        # gradient w.r.t. weights and biases
        dW = (delta.T @ layer.input) / batch_size   # (n_i, n_{i-1})
        db = delta.mean(axis=0, keepdims=True).T     # (n_i, 1)

        # propagate error signal to the layer below before touching weights
        if i > 1:
            delta = (layer.weights.T @ delta.T).T * network[i - 1].activation_gradient()

        layer.weights -= learning_rate * dW
        layer.biases  -= learning_rate * db


def fit(network, X_train, y_train, X_val, y_val,
        loss="categoricalCrossentropy", learning_rate=0.01,
        batch_size=32, epochs=100):
    """
    Train the network using mini-batch gradient descent.
    Returns a history dict with loss/accuracy per epoch (for plotting).
    """
    if loss != "categoricalCrossentropy":
        raise ValueError(f"unsupported loss '{loss}' — only 'categoricalCrossentropy' is supported for training")

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    n = len(X_train)

    for epoch in range(1, epochs + 1):
        # shuffle training data each epoch
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        # mini-batch passes
        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            out = forward(network, Xb)
            _backward(network, out, yb, learning_rate)

        # full-dataset metrics for reporting (no gradient update here)
        train_out = forward(network, X_train)
        val_out   = forward(network, X_val)

        train_loss = categorical_cross_entropy(train_out, y_train)
        val_loss   = categorical_cross_entropy(val_out,   y_val)
        train_acc  = compute_accuracy(train_out, y_train)
        val_acc    = compute_accuracy(val_out,   y_val)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    _plot_history(history)

    return history


def _plot_history(history):
    """Display loss and accuracy learning curves after training."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["loss"],     label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(epochs, history["acc"],     label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def save_model(network, mean, std, classes, filepath="saved_model.npy"):
    """
    Serialize the network to a single .npy file.
    Stores topology, weights, biases, and preprocessing params so
    predict.py can reconstruct and apply the exact same pipeline.
    """
    topology = []
    weights  = []
    biases   = []

    for layer in network:
        topology.append({
            "n_neurons":           layer.n_neurons,
            "activation":          layer.activation,
            "weights_initializer": layer.weights_initializer,
        })
        weights.append(layer.weights)
        biases.append(layer.biases)

    model_data = {
        "topology": topology,
        "weights":  weights,
        "biases":   biases,
        "mean":     mean,
        "std":      std,
        "classes":  classes,
    }

    np.save(filepath, model_data)
    print(f"> saving model '{filepath}' to disk...")


def load_model(filepath="saved_model.npy"):
    """
    Reconstruct a network from a saved .npy file.
    Returns (network, mean, std, classes).
    """
    data = np.load(filepath, allow_pickle=True).item()

    network = []
    for i, t in enumerate(data["topology"]):
        layer = DenseLayer(t["n_neurons"], t["activation"], t["weights_initializer"])
        layer.weights = data["weights"][i]
        layer.biases  = data["biases"][i]
        network.append(layer)

    return network, data["mean"], data["std"], data["classes"]
