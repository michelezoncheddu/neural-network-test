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
        error_out = 0
        for pattern in training_set:
            # Compute input layer without class attribute.
            outputs = list(map(self.activation_function, pattern[1:]))

            for layer in self.layers:  # Compute inner layers.
                outputs = layer.compute(outputs)  # Outputs of the previous layer are given to the current.

            error_out += pattern[0] - outputs[0]  # NOTE: One output unit only.

        # Mean error.
        error_out /= len(training_set)

        # Derivative of logistic function.
        derivative = lambda x: math.exp(x) / math.pow(1 + math.exp(x), 2)

        # Backpropagation.

        # Array of 𝛿k (output units).
        delta_outputs = []
        for unit in self.layers[-1].units:
            delta_outputs.append(error_out * derivative(unit.net))

        # Number of inputs of a generic output unit:
        # note that the network is fully-connected.
        n_inputs = len(self.layers[-1].units[0].weights)
        DELTA_W = [0] * n_inputs  # Total gradient.

        # Output layer gradient computation (step 1 on slides).
        for t in range(len(self.layers[-1].units)):  # For every output unit t.

            # Δ Wt (gradient for the mean error of the unit t).
            DELTA_Wt = []

            # For every input i in unit t.
            for i in range(n_inputs):
                DELTA_Wt.append(delta_outputs[t] * self.layers[-2].units[i].output)

            DELTA_W = [x + y for x, y in zip(DELTA_W, DELTA_Wt)]  # Vectorial sum.
