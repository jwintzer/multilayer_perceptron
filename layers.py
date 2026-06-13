import numpy as np


# --- activation functions ---

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    # subtract row-wise max for numerical stability
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


# --- layer ---

class DenseLayer:
    def __init__(self, n_neurons, activation="sigmoid", weights_initializer=None):
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights_initializer = weights_initializer

        self.weights = None
        self.biases = None

        # stored during forward pass, needed by backprop
        self.input = None   # X coming into this layer
        self.z = None       # pre-activation: W @ X.T + b
        self.output = None  # post-activation

    def init_weights(self, n_inputs):
        """
        Initialize W (n_neurons, n_inputs) and b (n_neurons, 1).
        Called by model.createNetwork() once the previous layer size is known.
        """
        if self.weights_initializer == "heUniform":
            limit = np.sqrt(6.0 / n_inputs)  # He uniform: sample from Uniform(-limit, limit)
            self.weights = np.random.uniform(-limit, limit, (self.n_neurons, n_inputs))
        elif self.weights_initializer == "xavier":
            limit = np.sqrt(6.0 / (n_inputs + self.n_neurons))
            self.weights = np.random.uniform(-limit, limit, (self.n_neurons, n_inputs))
        else:
            self.weights = np.random.randn(self.n_neurons, n_inputs) * 0.01

        self.biases = np.zeros((self.n_neurons, 1))

    def forward(self, X):
        """
        Forward pass through this layer.
        X    : (batch_size, n_inputs)
        out  : (batch_size, n_neurons)

        The input layer has no weights — it just passes X through unchanged.
        """
        if self.weights is None:
            self.output = X
            return X

        self.input = X
        # W: (n_neurons, n_inputs), X.T: (n_inputs, batch_size)
        # z: (batch_size, n_neurons) after transposing
        self.z = (self.weights @ X.T + self.biases).T

        if self.activation == "sigmoid":
            self.output = sigmoid(self.z)
        elif self.activation == "softmax":
            self.output = softmax(self.z)
        else:
            self.output = self.z  # linear, shouldn't normally be used here

        return self.output

    def activation_gradient(self):
        """
        Derivative of the activation w.r.t. z (pre-activation).
        Used during backprop. Softmax gradient is handled separately
        (combined with the cross-entropy loss for numerical stability).
        """
        if self.activation == "sigmoid":
            s = sigmoid(self.z)
            return s * (1.0 - s)
        # softmax: caller handles it via the loss gradient directly
        return np.ones_like(self.z)
