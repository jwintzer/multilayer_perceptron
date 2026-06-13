import numpy as np
import pandas as pd


def load_data(filepath, label_col=1):
    """
    Load a CSV dataset.
    Assumes col 0 is a sample ID (always dropped).
    Returns X (float features) and y_raw (raw string labels).
    """
    df = pd.read_csv(filepath, header=None)

    # drop sampleID column
    df = df.drop(columns=[0]).reset_index(drop=True)

    # after dropping col 0, label_col shifts left by 1
    label_idx = label_col - 1
    y_raw = df.iloc[:, label_idx].values
    X = df.drop(columns=[df.columns[label_idx]]).values.astype(float)

    return X, y_raw


def encode_labels(y_raw, classes=None):
    """
    One-hot encode string labels.
    Classes are sorted alphabetically so the mapping is deterministic:
      B -> index 0 -> [1, 0]
      M -> index 1 -> [0, 1]

    Pass classes explicitly to reuse the ordering from the training set
    (so val/test never derive their own ordering).

    Returns:
        y_onehot : (n_samples, n_classes) float array
        classes  : sorted list of class names (index = class id)
    """
    if classes is None:
        classes = sorted(set(y_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y = np.zeros((len(y_raw), len(classes)))
    for i, label in enumerate(y_raw):
        y[i, class_to_idx[label]] = 1.0

    return y, classes


def to_binary_labels(y_raw, classes):
    """
    Convert raw labels to binary int array using the class index.
    Used by predict.py for binary cross-entropy evaluation.
      B -> 0, M -> 1
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return np.array([class_to_idx[label] for label in y_raw])


def normalize(X, mean=None, std=None):
    """
    Z-score normalization.
    If mean/std are None, compute them from X (training set).
    Returns (X_norm, mean, std) so the caller can store and reuse them.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    # constant features would cause div-by-zero — clamp to 1
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std
