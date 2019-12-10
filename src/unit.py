
class Unit:
    """The network's elementary elaboration unit."""

    def __init__(self):
        self.weights = []
        self.bias = 0.05
        self.output = 0
    
    def compute(self, activation_function, inputs):
        self.net = 0  # Partial result.
        weight_index = 0

        # Compute the net value adding together the inputs
        # multiplied by their weights.
        for data in inputs:
            self.net += float(data) * self.weights[weight_index]
            weight_index += 1

        self.net += self.bias
        self.output = activation_function(self.net)
        return self.output
