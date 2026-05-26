# Multilayer Perceptron

A multilayer perceptron for binary classification, implemented from scratch in Python.

No ML libraries — forward pass, backpropagation, and gradient descent are all written manually. NumPy handles linear algebra, Matplotlib handles plots.

---

## Overview

The network reads a CSV dataset, separates it into training and validation sets, trains on the training set, and evaluates on the validation set. Input shape, output shape, and hidden layer topology are all inferred or configured at runtime — the implementation is not tied to any specific dataset.

The default use case is the Wisconsin Breast Cancer dataset (569 samples, 30 features, binary label: M/B), but any CSV with a categorical label column will work.

---

## Usage

### 1. Split the dataset

```bash
python split.py data.csv
```

Options:
- `--ratio` / `-r` : fraction of data used for training (default: `0.8`)
- `--seed` / `-s` : random seed for reproducibility
- `--label-col` : index of the label column (default: `1`)

Outputs `train.csv` and `val.csv`.

### 2. Train

```bash
python train.py train.csv val.csv
```

Options:
- `--layer` : hidden layer sizes, space-separated (default: `24 24`)
- `--epochs` : number of training epochs (default: `100`)
- `--loss` : loss function — `categoricalCrossentropy` or `binaryCrossentropy` (default: `categoricalCrossentropy`)
- `--learning-rate` : learning rate (default: `0.01`)
- `--batch-size` : mini-batch size (default: `32`)
- `--weights-initializer` : weight init method — `heUniform` or `xavier` (default: `heUniform`)

Prints loss and val_loss at each epoch. Saves model to `saved_model/` at the end.

Example:
```bash
python train.py train.csv val.csv --layer 24 24 --epochs 84 --loss categoricalCrossentropy --batch-size 8 --learning-rate 0.0314
```

### 3. Predict

```bash
python predict.py val.csv
```

Options:
- `--model` / `-m` : path to saved model directory (default: `saved_model`)

Loads the saved model, runs inference, and evaluates using binary cross-entropy.

---

## Network architecture

- **Input layer**: size inferred from dataset features
- **Hidden layers**: configurable count and size, sigmoid activation, He uniform init
- **Output layer**: size inferred from number of classes, softmax activation

---

## Requirements

```bash
pip install -r requirements.txt
```
