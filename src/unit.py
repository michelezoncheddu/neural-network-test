
class Unit:
    def __init__(self):
        self.weights = []
        self.bias = 0.5

    def compute(self, activation_function, inputs):
        value = 0
        index = 0
        for data in inputs:
            value += int(data) * self.weights[index]
            index += 1

        return activation_function(value)
