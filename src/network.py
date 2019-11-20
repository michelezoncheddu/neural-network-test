import math
import random

from layer import Layer


class Network:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Init a neural network with:
           - num_inputs input units
           - 1 hidden layer with num_hidden units
           - num_outputs output units
        """

        self.num_inputs = num_inputs
        
        # Sigmoidal logistic function.
        self.activation_function = lambda x: 1 / (1 + math.exp(-x))

        self.layers = []
        self.layers.append(Layer(num_inputs, self.activation_function))  # Temporary input layer.
        self.layers.append(Layer(num_hidden, self.activation_function))
        self.layers.append(Layer(num_outputs, self.activation_function))

        for i in range(len(self.layers)):
            self.init_layer_weights(self.layers[i], self.layers[i - 1])
        
        self.layers.pop(0)

    def init_layer_weights(self, layer, previous_layer):
        """Init layer weights with random values"""
        for unit in layer.units:
            for _ in range(previous_layer.num_units()):
                unit.weights.append(random.random() * 10)  # [0, 0.1)

    def train(self, training_set):
        for pattern in training_set:
            outputs = map(self.activation_function, pattern)  # Compute input layer.
            for layer in self.layers:
                outputs = layer.compute(outputs)
            
            # Test.
            print(outputs)
            break
