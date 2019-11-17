from unit import Unit


class Layer:
    def __init__(self, num_units):
        self.units = []

        for i in range(num_units):
            self.units.append(Unit())

    def compute(self, activation_function, inputs):
        outputs = []
        for unit in self.units:
            outputs.append(unit.compute(activation_function, inputs))
        return outputs
