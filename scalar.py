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
        self.gradient = 1

        next_visit = list(reversed(self.topological_order(self)))

        # find d(out) / d(self) and d(out) / d(other)
        for cur in next_visit:
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

    def topological_order(self, cur, visited=[], out=[]):
        if cur not in visited:
            visited.append(cur)
            if len(cur.parents) != 0:
                self.topological_order(cur.parents[0])
                self.topological_order(cur.parents[1])
            out.append(cur)
        return out

    def graph(self):
        G = nx.DiGraph()
        self.generate_graph(self, G)
        print('Max idx is: ' + str(idx - 1))
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
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a)
d += 3 * d + (b - a)
print(c.value)
print('---------')
d.backward()
print(a.gradient)
print(b.gradient)
# print(c.gradient)
c.graph()
# print(d.gradient)
# print(e.gradient)
# print(f.gradient)
