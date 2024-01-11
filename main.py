import numpy as np


class Tensor:
    def __init__(self, inputs):
        self.tensor = np.asarray(inputs)