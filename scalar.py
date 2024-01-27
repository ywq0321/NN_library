import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

idx = 0


class Value:
    def __init__(self, value, parents=(), label=0):
        if type(value) == int or type(value) == float:
            self.value = value
        else:
            raise ValueError('int or float expected, but ' + str(type(value)) + ' is given. ')
        self.parents = parents
        self.gradient = 0
        global idx
        self.idx = idx
        idx += 1

    def __add__(self, other):
        other = other if type(other) == Value else Value(other)
        return Value(self.value + other.value, (self, other, 'add'))

    def __mul__(self, other):
        other = other if type(other) == Value else Value(other)
        return Value(self.value * other.value, (self, other, 'mul'))

    def __pow__(self, other):
        other = other if type(other) == Value else Value(other)
        return Value(self.value ** other.value, (self, other, 'pow'))

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def relu(self):
        return self.value if self.value > 0 else 0

    def backward(self):
        next_visit = [self]
        cur = self
        self.gradient = 1
        history = []

        # find d(out) / d(self) and d(out) / d(other)
        while 1:
            if len(next_visit) == 0:
                break
            cur = next_visit.pop(0)
            if cur.idx in history:
                continue
            history.append(cur.idx)
            print(history)
            if cur.parents == ():
                continue


            if cur.parents[2] == 'add':
                cur.parents[0].gradient += cur.gradient
                cur.parents[1].gradient += cur.gradient
            elif cur.parents[2] == 'mul':
                cur.parents[0].gradient += cur.gradient * cur.parents[1].value
                cur.parents[1].gradient += cur.gradient * cur.parents[0].value
            elif cur.parents[2] == 'pow':
                cur.parents[0].gradient += (cur.gradient *
                                            cur.parents[0].value ** (cur.parents[1].value - 1) * cur.parents[1].value)
                cur.parents[1].gradient += (cur.gradient *
                                            np.log(cur.parents[0].value) * cur.parents[0].value ** cur.parents[1].value)

            if cur.parents[0].idx > cur.parents[1].idx:
                next_visit.append(cur.parents[0])
                next_visit.append(cur.parents[1])
            else:
                next_visit.append(cur.parents[1])
                next_visit.append(cur.parents[0])

    def graph(self):
        G = nx.DiGraph()
        self.generate_graph(self, G)
        print('Max idx is: '+str(idx-1))
        nx.draw_networkx(G, arrows=True)
        plt.show()

    def generate_graph(self, cur, G):
        if cur.parents == ():
            return
        G.add_edge(cur.idx, cur.parents[0].idx, color='blue')
        G.add_edge(cur.idx, cur.parents[1].idx, color='red')
        self.generate_graph(cur.parents[0], G)
        self.generate_graph(cur.parents[1], G)


a = Value(-4.0)
b = Value(2.0)
c = a + b
# d = a * b + b**3
c += c + 1
# c += 1 + c + (-a)
# d += d * 2 + (b + a)
# d += 3 * d + (b - a)
# e = c - d
# f = e**2
# g = f / 2.0
# g += 10.0 / f
c.backward()
print(a.gradient)
print(b.gradient)
print(c.gradient)
c.graph()
print(d.gradient)
print(e.gradient)
print(f.gradient)
