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

        self.num_inputs = size[0]
        self.num_hidden = size[1]
        self.num_outputs = size[2]

        value = 0.01
        self.weights = [[np.random.uniform(-value, value) for _ in range(size[i - 1])] for i in range(1, len(size))]
        self.biases = [[np.random.uniform(-value, value) for _ in range(size[i])] for i in range(len(size))]
        
        self.nets = [np.zeros(size[i]) for i in range(1, len(size))]
        self.outputs = [np.empty(size[i]) for i in range(1, len(size))]

        self.delta = [np.empty(size[i]) for i in range(1, len(size))]
        self.gradients = [np.zeros(size[i]) for i in range(1, len(size))]
    
    @staticmethod
    def activation_function(x):
        """Sigmoidal logistic function"""
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative(x):
        """Derivative of sigmoidal function"""
        return np.exp(x) / math.pow(1 + np.exp(x), 2)
    
    def feedforward(self):


    def train(self, training_set):
        square_error = 0
        misclassifications = 0

        for pattern in training_set:
            # Array of 𝛿k (output and hidden units).
            delta_outputs = []
            delta_hidden = []

            # Compute input layer without class attribute.
            outputs = pattern[1:]

            # Output units deltas.
            for output_unit in self.layers[-1].units:
                error_out = pattern[0] - output_unit.output
                square_error += math.pow(pattern[0] - output_unit.output, 2)
                misclassifications += pattern[0] - round(output_unit.output)
                delta_outputs.append(error_out * self.derivative(output_unit.net))

            # Output layer gradient computation (step 1 on slides).
            for t in range(self.num_outputs):  # For every output unit t.
                # For every input i in output unit t.
                for i in range(self.num_hidden):
                    output_gradient[t][i] += delta_outputs[t] * self.layers[-2].units[i].output

            # Hidden units deltas.
            for h in range(self.num_hidden):
                delta_tmp = 0
                for o in range(self.num_outputs):
                    delta_tmp += delta_outputs[o] * self.layers[-1].units[o].weights[h]
                delta_tmp *= self.derivative(self.layers[-2].units[h].net)
                delta_hidden.append(delta_tmp)

            # Hidden layer gradient computation (step 1 on slides).
            for h in range(self.num_hidden):
                # For every input i in hidden unit h.
                for i in range(self.num_inputs):
                    hidden_gradient[h][i] += delta_hidden[h] * input_layer_outputs[i]

        # Output layer weights update.
        for o in range(self.num_outputs):
            self.layers[-1].units[o].bias -= self.LEARNING_RATE * delta_outputs[o]
            for i in range(self.num_hidden):
                self.layers[-1].units[o].weights[i] += self.LEARNING_RATE * output_gradient[o][i]
        
        # Hidden layer weights update.
        for h in range(self.num_hidden):
            self.layers[-2].units[h].bias -= self.LEARNING_RATE * delta_hidden[h]
            for i in range(self.num_inputs):
                self.layers[-2].units[h].weights[i] += self.LEARNING_RATE * hidden_gradient[h][i]

        print(square_error, misclassifications)

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.compute(outputs)
        return outputs
