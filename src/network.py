import math
import random

from layer import Layer


class Network:
    """Fully-connected feedforward neural network with one hidden layer."""

    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Init a neural network with:
           - num_inputs input units
           - 1 hidden layer with num_hidden units
           - num_outputs output units.
        """
        
        # Sigmoidal logistic function.
        self.activation_function = lambda x: 1 / (1 + math.exp(-x))

        self.layers = []
        self.layers.append(Layer(num_inputs, self.activation_function))  # Temporary input layer.
        self.layers.append(Layer(num_hidden, self.activation_function))
        self.layers.append(Layer(num_outputs, self.activation_function))

        # Initialize layers weights except the input one.
        for i in range(1, len(self.layers)):
            self.init_layer_weights(self.layers[i], self.layers[i - 1])
        
        self.layers.pop(0)  # Remove input layer.

    def init_layer_weights(self, layer, previous_layer):
        """Init layer weights with random values."""
        for unit in layer.units:
            for _ in range(len(previous_layer.units)):
                unit.weights.append(random.random() / 10)  # [0, 0.1)

    def train(self, training_set):
        for pattern in training_set:
            outputs = list(map(self.activation_function, pattern)) # Compute input layer.
            for layer in self.layers:  # Compute inner layers.
                outputs = layer.compute(outputs)  # Outputs of the previous layer are given to the current.

            # Test.
            print(outputs)
            break
