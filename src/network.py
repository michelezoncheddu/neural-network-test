import math
import numpy as np
import random


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
            [
                [
                    np.random.uniform(-value, value) for _ in range(size[i - 1])  # For every input of the previous layer.
                ]
                for _ in range(size[i])  # For every unit.
            ]
            for i in range(1, len(size))  # For every layer, without the input one.
        ]

        self.biases = [[np.random.uniform(-value, value) for _ in range(size[i])] for i in range(1, len(size))]
        
        self.nets =    [np.zeros(size[i]) for i in range(1, len(size))]
        self.outputs = [np.empty(size[i]) for i in range(1, len(size))]

        self.deltas =  [np.empty(size[i]) for i in range(1, len(size))]
    
    @staticmethod
    def activation_function(x):
        """Sigmoidal logistic function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        """Derivative of sigmoidal function"""
        return np.exp(x) / math.pow(1 + np.exp(x), 2)
    
    def feedforward(self, x):
        activation_function = np.vectorize(self.activation_function)
        for i in range(len(self.weights)):  # For every layer.
            self.nets[i] = np.dot(
                self.weights[i],
                x if i == 0 else self.outputs[i - 1]) + self.biases[i]
            self.outputs[i] = activation_function(self.nets[i])

    def backpropagation(self, training_set):
        square_error = 0
        misclassifications = 0

        """gradients = [
            [
                np.zeros(range(self.size[i - 1])) for _ in range(self.size[i])  # For every unit.
            ]
            for i in range(1, len(self.size))  # For every layer.
        ]"""

        gradients = [
            [
                [
                    0 for _ in range(self.size[i - 1])  # For every input of the previous layer.
                ]
                for _ in range(self.size[i])  # For every unit.
            ]
            for i in range(1, len(self.size))  # For every layer, without the input one.
        ]

        derivative = np.vectorize(self.derivative)

        for pattern in training_set:
            self.feedforward(pattern[1:])

            # Output layer deltas.
            """self.deltas[-1] = np.multiply(
                np.subtract(
                    self.outputs[-1],
                    pattern[:1]),
                derivative(self.nets[-1]))

            # Hidden units deltas.
            for i in reversed(range(len(self.weights) - 1)):
                self.deltas[i] = np.multiply(
                    np.dot(
                        self.deltas[i + 1],
                        np.sum(self.weights[i + 1], axis=1)),
                    derivative(self.nets[i]))

            # Gradient computation.
            for i in reversed(range(len(self.size) - 1)):
                for j in range(len(gradients[i])):
                    gradients[i][j] = np.multiply(
                        self.deltas[i][j],
                        pattern[1:] if i == 0 else self.outputs[i - 1])

            # Weights update.
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] += gradients[i][j][k]"""

        print(square_error, misclassifications)

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.compute(outputs)
        return outputs
