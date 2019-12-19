import math
import numpy as np
import random
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

        self.biases =  [np.random.uniform(-value, value, size[i]) for i in range(1, len(size))]
        
        self.nets =    [np.zeros(size[i]) for i in range(1, len(size))]
        self.outputs = [np.empty(size[i]) for i in range(1, len(size))]

        self.deltas =  [np.empty(size[i]) for i in range(1, len(size))]
    
    @staticmethod
    def activation_function(x):
        """Sigmoidal logistic function"""
        return expit(x)
    
    @staticmethod
    def derivative(x):
        """Derivative of sigmoidal function (using the differential equation)"""
        fx = expit(x)
        return fx * (1.0 - fx)
    
    def feedforward(self, x):
        for i in range(len(self.weights)):  # For every layer.
            self.nets[i] = np.dot(
                self.weights[i],
                x if i == 0 else self.outputs[i - 1]
            ) + self.biases[i]
            self.outputs[i] = self.activation_function(self.nets[i])

    def backpropagation(self, training_set):
        # TODO: update bias

        square_error = 0
        misclassifications = 0

        for pattern in training_set:
            gradients = [
                np.zeros((self.size[i], self.size[i - 1]))
                for i in range(1, len(self.size))  # For every layer, without the input one.
            ]

            self.feedforward(pattern[1:])

            error = pattern[:1] - self.outputs[-1]
            square_error += math.pow(np.sum(error), 2)

            # Output layer deltas.
            self.deltas[-1] = error * self.derivative(self.nets[-1])

            # Hidden units deltas.
            for i in reversed(range(len(self.weights) - 1)):
                self.deltas[i] = np.dot(
                    self.deltas[i + 1],
                    self.weights[i + 1]
                ) * self.derivative(self.nets[i])

            # NOTE: optimize below

            # Gradient computation.
            for i in reversed(range(len(self.weights))):
                for j in range(len(gradients[i])):
                    gradients[i][j] = np.multiply(
                        self.deltas[i][j],
                        pattern[1:] if i == 0 else self.outputs[i - 1])

            # Weights update.
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] += self.LEARNING_RATE * gradients[i][j][k]

        print(square_error)

    def predict(self, inputs):
        self.feedforward(inputs)
        return self.outputs[-1]
