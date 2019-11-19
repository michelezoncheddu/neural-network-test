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

        self.hidden_layer = Layer(num_hidden)
        self.output_layer = Layer(num_outputs)

        self.init_hidden_layer_weights()
        self.init_output_layer_weights()
        
    def init_hidden_layer_weights(self):
        """Init hidden layer weights with random values"""
        for h in range(len(self.hidden_layer.units)):
            for _ in range(self.num_inputs):
                self.hidden_layer.units[h].weights.append(random.random() / 10)  # [0, 0.1)

    def init_output_layer_weights(self):
        """Init output layer weights with random values"""
        for o in range(len(self.output_layer.units)):
            for _ in range(len(self.hidden_layer.units)):
                self.output_layer.units[o].weights.append(random.random() / 10)  # [0, 0.1)

    def train(self, training_set):
        # Layers functions.
        input_layer_function = lambda x: 2 * x
        hidden_layer_function = lambda x: 2 * x
        output_layer_function = lambda x: 3 * x

        for data in training_set:

            # Compute input layer functions.
            input_layer_outputs = []
            for i in range(self.num_inputs):
                input_layer_outputs.append(input_layer_function(data[i]))
            
            
            hidden_layer_outputs = self.hidden_layer.compute(hidden_layer_function, input_layer_outputs)
            self.output_layer.compute(output_layer_function, hidden_layer_outputs)
