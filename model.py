import numpy as np


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
