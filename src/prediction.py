
class Prediction:
    """A simple class of couple < mse, label >"""
    def __init__(self, square_error, label):
        self.square_error = square_error
        self.label = label
  