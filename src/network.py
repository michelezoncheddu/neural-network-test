import math
import numpy as np
from scipy.special import expit


class Network:
    """Fully-connected feedforward neural network with one hidden layer."""

    LEARNING_RATE = 0.5

    def __init__(self, size):
        """Init a neural network with:
           - num_inputs input units
           - 1 hidden layer with num_hidden units
           - num_outputs output units.
        """

        self.size = size
        self.num_inputs = size[0]
        self.num_hidden = size[1]
        self.num_outputs = size[2]

        value = 0.01

        self.weights = [
            np.random.uniform(-value, value, (size[i], size[i - 1]))
            for i in range(1, len(size))  # For every layer, without the input one.
        ]

        self.biases = [np.random.uniform(-value, value, size[i]) for i in range(1, len(size))]

        self.nets = [np.zeros(size[i]) for i in range(1, len(size))]
        self.outputs = [np.empty(size[i]) for i in range(1, len(size))]

        # For backpropagation.
        self.deltas = []
        self.gradients = []

    @staticmethod
    def activation_function(x):
        """Sigmoidal logistic function"""
        return expit(x)

    @staticmethod
    def derivative(x):
        """Derivative of sigmoidal function (using the differential equation)"""
        f_x = expit(x)
        return f_x * (1.0 - f_x)

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
        deltas = [np.empty(self.size[i]) for i in range(1, len(self.size))]

        # Output layer deltas.
        deltas[-1] = error * self.derivative(self.nets[-1])

        # Hidden units deltas.
        for i in reversed(range(len(self.weights) - 1)):
            deltas[i] = np.dot(
                deltas[i + 1],
                self.weights[i + 1]
            ) * self.derivative(self.nets[i])

        # Gradient computation.
        for i in reversed(range(len(self.weights))):
            self.gradients[i] += deltas[i].reshape(-1, 1) \
                * (inputs if i == 0 else self.outputs[i - 1])
            self.deltas[i] += deltas[i]

    def train(self, training_set):
        """Trains the neural network (batch mode)."""
        square_error = 0

        self.deltas = [np.zeros(self.size[i]) for i in range(1, len(self.size))]

        self.gradients = [
            np.zeros((self.size[i], self.size[i - 1]))
            for i in range(1, len(self.size))  # For every layer, without the input one.
        ]

        for pattern in training_set:
            # TODO: 1 needs to be parameterized
            inputs = pattern[1:]
            targets = pattern[:1]

            self.forward_propagation(inputs)

            error = targets - self.outputs[-1]
            square_error += math.pow(np.sum(error), 2)

            self.back_propagation(inputs, error)

        # Bias and weights update.
        for i in range(len(self.weights)):
            self.weights[i] += self.LEARNING_RATE * self.gradients[i]
            self.biases[i] += self.LEARNING_RATE * self.deltas[i]

        print(square_error)

    def predict(self, inputs):
        """Calculates the neural network output."""
        self.forward_propagation(inputs)
        return self.outputs[-1]
