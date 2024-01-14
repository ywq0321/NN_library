import numpy as np


class DenseLayer:
    def __init__(self, width, out, acti='tanh'):
        self.width = width
        self.out = out
        self.acti = acti
        self.value = np.array([None for i in range(width)])
        self.weights = np.array([[np.random.rand() * 2 - 1 for i in range(out)] for i in range(width)])
        self.bias = np.array([np.random.rand() * 2 - 1 for i in range(out)])

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
    def __init__(self, widths: list):
        self.depth = len(widths) - 1
        self.widths = widths

        self.network = []
        for i in range(self.depth - 1):
            self.network.append(DenseLayer(widths[i], widths[i + 1]))
        self.network.append(DenseLayer(widths[self.depth - 1], widths[self.depth], acti='sigmoid'))

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
        sum = []
        for i in range(len(outputs)):
            sum.append(2 * np.abs(outputs[i] - labels[i]))
        return sum

    def transpose(self, x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        else:
            x = np.transpose(x)
        return x

    def back_propagate(self, error):
        error = np.array(error)
        learning_rate = 0.1

        for i in range(len(self.network) - 1, -1, -1):
            input_error = np.dot(error, self.transpose(self.network[i].weights))
            self.network[i].weights -= self.transpose(learning_rate * np.dot(self.network[i].value.reshape(self.widths[i], 1),
                                                                             error.reshape(1, self.widths[i+1])))
            self.network[i].bias -= learning_rate * error
            error = input_error

    def train(self, epochs: int, inputs: list, labels: list):
        assert len(inputs) == len(labels)
        assert len(inputs[0]) == self.widths[0]
        for i in range(epochs):
            print('Epoch:', i+1)
            error = 0
            error_dash = [0 for i in range(len(labels[0]))]
            for j in range(len(inputs)):
                predicted = self.forward_propagate(inputs[j])
                error += self.MSE(predicted, labels[j])
                err = self.MSE_dash(predicted, labels[j])
                for i in range(len(err)):
                    error_dash[i] += err[i]
            print('Error:', error)
            self.back_propagate(error_dash)
            # print(self.forward_propagate([1, 1]))
            # print(self.forward_propagate([1, 0]))
            # print(self.forward_propagate([0, 1]))
            # print(self.forward_propagate([0, 0]))


def square(x):
    return x ** 2


def generate_data(func, min, max, size):
    x = np.random.uniform(min, max, size)
    y = []
    for i in x:
        y.append([func(i)])
    return x.reshape((size, 1)), y


# x, y = generate_data(square, -10, 10, 20)
my_NN = NN([2, 2, 2])
my_NN.train(100, [[1, 1], [1, 0], [0, 1], [0, 0]], [[1, 1], [1, 0], [0, 1], [0, 0]])
