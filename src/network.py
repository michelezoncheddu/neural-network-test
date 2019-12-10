import math
import numpy as np
import random

from layer import Layer


class Network:
    """Fully-connected feedforward neural network with one hidden layer."""

    LEARNING_RATE = 0.2

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Init a neural network with:
           - num_inputs input units
           - 1 hidden layer with num_hidden units
           - num_outputs output units.
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

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
                unit.weights.append(np.random.uniform(-0.01, 0.01))

    def train(self, training_set):
        # Derivative of logistic function.
        derivative = lambda x: math.exp(x) / math.pow(1 + math.exp(x), 2)

        square_error = 0
        misclassifications = 0

        for pattern in training_set:
            # Array of 𝛿k (output and hidden units).
            delta_outputs = []
            delta_hidden = []

            # Compute input layer without class attribute.
            #outputs = list(map(self.activation_function, pattern[1:]))
            outputs = pattern[1:]
            input_layer_outputs = outputs.copy()

            for layer in self.layers:  # Compute inner layers.
                outputs = layer.compute(outputs)  # Outputs of the previous layer are given to the current.

            # Output units deltas.
            for output_unit in self.layers[-1].units:
                error_out = pattern[0] - output_unit.output
                square_error += math.pow(pattern[0] - output_unit.output, 2)
                misclassifications += pattern[0] - round(output_unit.output)
                delta_outputs.append(error_out * derivative(output_unit.net))

            # Output layer gradient computation (step 1 on slides).
            for t in range(self.num_outputs):  # For every output unit t.
                # For every input i in output unit t.
                for i in range(self.num_hidden):
                    self.layers[-1].units[t].weights[i] += self.LEARNING_RATE * (delta_outputs[t] * self.layers[-2].units[i].output)

            # Hidden units deltas.
            for h in range(self.num_hidden):
                delta_tmp = 0
                for o in range(self.num_outputs):
                    delta_tmp += delta_outputs[o] * self.layers[-1].units[o].weights[h]
                delta_tmp *= derivative(self.layers[-2].units[h].net)
                delta_hidden.append(delta_tmp)

            # Hidden layer gradient computation (step 1 on slides).
            for h in range(self.num_hidden):
                # For every input i in hidden unit h.
                for i in range(self.num_inputs):
                    self.layers[-2].units[h].weights[i] += self.LEARNING_RATE * (delta_hidden[h] * input_layer_outputs[i])

        print(misclassifications)
