
class Unit:
    """The network's elementary elaboration unit."""

    def __init__(self):
        self.weights = []
        self.bias = 0.5
    
    def compute(self, activation_function, inputs):
        net = 0  # Partial result.
        weight_index = 0

        # Compute the net value adding together the inputs
        # multiplied by their weights.
        for data in inputs:
            net += float(data) * self.weights[weight_index]
            weight_index += 1

        return activation_function(net + self.bias)
