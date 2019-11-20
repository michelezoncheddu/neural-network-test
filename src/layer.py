from unit import Unit


class Layer:
    def __init__(self, num_units, activation_function):
        self.units = []
        self.activation_function = activation_function

        for i in range(num_units):
            self.units.append(Unit())

    def num_units(self):
        return len(self.units)

    def compute(self, inputs):
        outputs = []
        for unit in self.units:
            outputs.append(unit.compute(self.activation_function, inputs))
        return outputs
