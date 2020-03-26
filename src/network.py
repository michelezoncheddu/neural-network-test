import math
import numpy as np
from scipy.special import expit


class Network:
    """A simple feedforward neural network."""

    LEARNING_RATE = 0.5
    ALPHA = 0.5  # Momentum hyperparameter.
    MINIBATCH = 0

    def __init__(self, size):
        """Init a neural network with:
           - num_inputs input units
           - 1 hidden layer with num_hidden units
           - num_outputs output units.
        """
        # TODO: size! - Michele
        self.size = size
        self.num_inputs = size[0]
        self.num_hidden = size[1]
        self.num_outputs = size[2]

        self.weights = []
        self.biases = []

        # Weights and biases initialization.
        # TODO: 6 needs to be parameterized - Michele
        for i in range(1, len(size)):  # For every layer, without the input one.
            value = math.sqrt(6 / (size[i - 1] + size[i]))
            self.weights.append(
                np.random.uniform(-value, value, (size[i], size[i - 1])))
            self.biases.append(np.zeros(size[i]))

        self.nets = [np.zeros(size[i]) for i in range(1, len(size))]
        self.outputs = [np.empty(size[i]) for i in range(1, len(size))]

        # For backpropagation.
        self.deltas_tmp = [np.empty(self.size[i]) for i in range(1, len(self.size))]

        self.deltas = []
        self.gradients = []
        self.weights_momentum = [
            np.zeros((self.size[i], self.size[i - 1]))
            for i in range(1, len(self.size))  # For every layer, without the input one.
        ]
        self.biases_momentum = [np.zeros(self.size[i]) for i in range(1, len(self.size))]

    @staticmethod
    def activation_function(x):
        """Sigmoidal logistic function."""
        return expit(x)  # RELU: x * (x > 0)

    @staticmethod
    def derivative(x):
        """Derivative of sigmoidal function (using the differential equation)."""
        f_x = expit(x)
        return f_x * (1.0 - f_x)  # RELU: 1. * (x >= 0)

    def forward_propagation(self, x):
        """Runs the neural network."""
        for i in range(len(self.weights)):  # For every layer.
            self.nets[i] = np.dot(
                self.weights[i],
                x if i == 0 else self.outputs[i - 1]
            ) + self.biases[i]
            self.outputs[i] = self.activation_function(self.nets[i])

    def back_propagation(self, inputs, error):
        """Performs the backpropagation algorithm."""

        # Output layer deltas.
        self.deltas_tmp[-1] = error * self.derivative(self.nets[-1])

        # Hidden units deltas.
        for i in reversed(range(len(self.weights) - 1)):
            self.deltas_tmp[i] = np.dot(
                self.deltas_tmp[i + 1],
                self.weights[i + 1]
            ) * self.derivative(self.nets[i])

        # Gradient computation.
        for i in reversed(range(len(self.weights))):
            self.gradients[i] += self.deltas_tmp[i].reshape(-1, 1) \
                * (inputs if i == 0 else self.outputs[i - 1])
            self.deltas[i] += self.deltas_tmp[i]

    def train(self, training_set):
        """Trains the neural network."""
        square_error = 0
        seen = 0
        misclassifications = 0
        target_dimension = 1

        # Needed to store bias "gradient" in batch mode.
        self.deltas = [np.zeros(self.size[i]) for i in range(1, len(self.size))]

        self.gradients = [
            np.zeros((self.size[i], self.size[i - 1]))
            for i in range(1, len(self.size))  # For every layer, without the input one.
        ]

        for pattern in training_set:
            seen += 1

            inputs = pattern[target_dimension:]
            targets = pattern[:target_dimension]

            self.forward_propagation(inputs)

            error = targets - self.outputs[-1]
            if round(self.outputs[-1][0]) != targets:  # Misclassification.
                misclassifications += 1
            square_error += math.pow(np.sum(error), 2)

            self.back_propagation(inputs, error)

            # TODO: update weights if len(ts) % MB != 0 - Michele
            if seen == self.MINIBATCH:
                seen = 0

                # Bias and weights update.
                for i in range(len(self.weights)):
                    self.weights_momentum[i] = \
                        self.LEARNING_RATE * self.gradients[i] / self.MINIBATCH \
                        + self.ALPHA * self.weights_momentum[i]
                    self.weights[i] += self.weights_momentum[i]

                    self.biases_momentum[i] = \
                        self.LEARNING_RATE * self.deltas[i] / self.MINIBATCH \
                        + self.ALPHA * self.biases_momentum[i]
                    self.biases[i] += self.biases_momentum[i]

                self.gradients = [
                    np.zeros((self.size[i], self.size[i - 1]))
                    for i in range(1, len(self.size))  # For every layer, without the input one.
                ]

                self.deltas = [np.zeros(self.size[i]) for i in range(1, len(self.size))]

        return square_error, misclassifications

    def predict(self, pattern):
        """Calculates the neural network output."""
        inputs = pattern[1:]
        targets = pattern[:1]
        square_error = 0
        self.forward_propagation(inputs)
        error = targets - self.outputs[-1]
        square_error = math.pow(np.sum(error), 2)
        return square_error, self.outputs[-1]
