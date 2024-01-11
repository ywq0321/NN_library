import numpy as np
from random import random


class Layer:
    def __init__(self, width, out):
        self.width = width
        self.out = out
        self.value = np.array([None for i in range(width)])
        self.weights = np.array([[random() for i in range(out)] for i in range(width)])
        self.bias = np.array([random() for i in range(out)])

    def __call__(self, *args, **kwargs):  # get output
        return np.dot(self.value, self.weights) + self.bias


class NN:
    def __init__(self, depth: int, widths: list):
        assert len(widths) == depth+1
        self.depth = depth
        self.widths = widths

        self.network = []
        for i in range(depth):
            self.network.append(Layer(widths[i], widths[i+1]))

    def forward_propagate(self, inputs):
        assert len(inputs) == self.widths[0]
        self.network[0].value = np.array(inputs)
        for i in range(self.depth-1):
            self.network[i+1].value = self.network[i]()
        return self.network[-1]()

    def back_propagate(self, inputs, labels):
        raise NotImplementedError


myNN = NN(3, [8, 4, 2, 1])
print(myNN.forward_propagate([i for i in range(8)]))
