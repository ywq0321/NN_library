import numpy as np


class DenseLayer:
    def __init__(self, width, out, activation='ReLU'):
        self.width = width
        self.out = out
        self.activation = activation
        self.value = np.array([None for i in range(width)])
        self.weights = np.array([[np.random.rand() * 2 - 1 for i in range(out)] for i in range(width)])
        self.bias = np.array([np.random.rand() * 2 - 1 for i in range(out)])

    def __call__(self):  # get output
        return self.activation(np.dot(self.value, self.weights) + self.bias, self.activation)

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
        assert len(widths) == depth + 1
        self.depth = depth
        self.widths = widths

        self.network = []
        for i in range(depth - 1):
            self.network.append(DenseLayer(widths[i], widths[i + 1]))
        self.network.append(DenseLayer(widths[depth - 1], widths[depth], activation='sigmoid'))

    def forward_propagate(self, inputs: list):
        assert len(inputs) == self.widths[0]
        self.inputs = inputs
        self.network[0].value = np.array(inputs)
        for i in range(self.depth - 1):
            self.network[i + 1].value = self.network[i]()
        return self.network[-1]()

    def MSE(self, outputs, labels):
        sum = 0
        for i in range(len(outputs)):
            sum += np.abs(outputs[i] ** 2 - labels[i] ** 2)
        return sum / len(outputs)

    def MSE_dash(self, outputs, labels):
        sum = 0
        for i in range(len(outputs)):
            sum += np.abs(outputs[i] - labels[i])
        return sum * 2 / len(outputs)

    def back_propagate(self, error):
        learning_rate = 0.01

        for i in range(len(self.network) - 1, -1, -1):
            input_error = np.dot(error, self.network[i].weights)
            self.network[i].weights -= learning_rate * np.dot(self.network[i].value, error)
            self.network[i].bias -= learning_rate * error
            error = input_error

    def train(self, epochs: int, inputs: list, labels: list):
        assert len(inputs) == len(labels)
        assert len(inputs) == self.widths[0]
        for i in range(epochs):
            print('Epoch:', i)
            for j in range(len(inputs)):
                predicted = self.forward_propagate(inputs[j])
                error = self.MSE(predicted, labels[i])
                error_dash = self.MSE_dash(predicted, labels[i])
                print('Error:', error)
                self.back_propagate(error_dash)


def square(x):
    return x ** 2


def generate_data(func, min, max, size):
    x = np.random.uniform(min, max, size)
    y = []
    for i in x:
        y.append(func(x))
    return y


