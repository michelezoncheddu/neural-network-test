from unit import Unit


class Layer:
    """Basic layer class."""

    def __init__(self, num_units, activation_function):
        self.units = []
        self.activation_function = activation_function

        for _ in range(num_units):
            self.units.append(Unit())

    def compute(self, inputs):
        """Computes the output for each unit of the layer."""
        outputs = []
        for unit in self.units:
            outputs.append(unit.compute(self.activation_function, inputs))

        return outputs
