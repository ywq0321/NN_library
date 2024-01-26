import numpy as np


class Value:
    def __init__(self, value, parents=()):
        if type(value) == int or type(value) == float:
            self.value = value
        else:
            raise ValueError('int or float expected, but ' + str(type(value)) + ' is given. ')
        self.parents = parents
        self.gradient = 0

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

    def backward(self, cur=None, visited=None):
        if visited is None:
            visited = []
        if cur is None:
            cur = self
            self.gradient = 1
        if cur.parents == ():
            print(visited)
            return

        # find d(out) / d(self) and d(out) / d(other)
        if cur.parents[2] == 'add':
            cur.parents[0].gradient += cur.gradient
            cur.parents[1].gradient += cur.gradient
            visited.append([cur.parents[0].value, cur.parents[1].value, 'add'])
        elif cur.parents[2] == 'mul':
            cur.parents[0].gradient += cur.gradient * cur.parents[1].value
            cur.parents[1].gradient += cur.gradient * cur.parents[0].value
            visited.append([cur.parents[0].value, cur.parents[1].value, 'mul'])
        elif cur.parents[2] == 'pow':
            cur.parents[0].gradient += (cur.gradient *
                                       cur.parents[0].value ** (cur.parents[1].value - 1) * cur.parents[1].value)
            cur.parents[1].gradient += (cur.gradient *
                                       cur.parents[1].value ** (cur.parents[0].value - 1) * cur.parents[0].value)
            visited.append([cur.parents[0].value, cur.parents[1].value, 'pow'])

        self.backward(cur=cur.parents[0], visited=visited)
        self.backward(cur=cur.parents[1], visited=visited)
        return


a = Value(-4.0)
b = Value(2.0)
c = a + b
# d = a * b + b**3
c = 1 + c + c
c = 1 + c + c + (-a)
# d += d * 2 + (b + a).relu()
# d += 3 * d + (b - a).relu()
# e = c - d
# f = e**2
# g = f / 2.0
# g += 10.0 / f
c.backward()
print(a.gradient)
print(b.gradient)
print(c.gradient)
print(d.gradient)
print(e.gradient)
print(f.gradient)
