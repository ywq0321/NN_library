import numpy as np
from random import random


class Layer:
    def __init__(self, width, out, acti='ReLU'):
        self.width = width
        self.out = out
        self.acti = acti
        self.value = np.array([None for i in range(width)])
        self.weights = np.array([[random()*2-1 for i in range(out)] for i in range(width)])
        self.bias = np.array([random()*2-1 for i in range(out)])

    def __call__(self, *args, **kwargs):  # get output
        return self.activation(np.dot(self.value, self.weights) + self.bias, self.acti)

    def activation(self, x, func):
        if func == 'ReLU':
            for i in range(len(x)):
                x[i] = x[i] if x[i] > 0 else 0
        elif func == 'tanh':
            for i in range(len(x)):
                x[i] = np.tanh(x[i])
        elif func == 'sigmoid':
            for i in range(len(x)):
                x[i] = 1 / (1 + np.exp(-x[i]))
        return x


class NN:  # creates a NN with ReLU activation except the last layer, which has sigmoid
    def __init__(self, depth: int, widths: list):
        assert len(widths) == depth+1
        self.depth = depth
        self.widths = widths

        self.network = []
        for i in range(depth-1):
            self.network.append(Layer(widths[i], widths[i+1]))
        self.network.append(Layer(widths[depth-1], widths[depth], acti='sigmoid'))

    def forward_propagate(self, inputs: list):
        assert len(inputs) == self.widths[0]
        self.network[0].value = np.array(inputs)
        for i in range(self.depth-1):
            self.network[i+1].value = self.network[i]()
        return self.network[-1]()

    def back_propagate(self, inputs: list, labels: list):
        raise NotImplementedError
        assert len(inputs) == self.widths[0]
        assert len(inputs) == self.widths[-1]


myNN = NN(3, [8, 4, 2, 1])
print(myNN.forward_propagate([i for i in range(8)])[0])
