from scalar import Value
import matplotlib.pyplot as plt

# initialise
A = -4.0
B = 2

# suppose you want to get g = 5
true = 5

epochs = 50
learning_rate = 0.000005
preds = []
losses = []

def MSE(pred, true):
    return (pred - true) ** 2

def MSE_diff(pred, true):
    return 2*(pred - true)


for i in range(epochs):
    a = Value(A)
    b = Value(B)
    c = a + b
    d = a * b + b ** 3
    c = c + 1 + c
    c = 1 + c + (-a) + c
    d = d * 2 + (b + a).relu() + d
    d = 3 * d + (b - a).relu() + d
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = 10.0 / f + g

    print(g.value)
    print(A, B)
    print()
    preds.append(g.value)
    losses.append(MSE(g.value, true))

    g.backward()
    A -= learning_rate * a.gradient * MSE_diff(g.value, true)
    B -= learning_rate * b.gradient * MSE_diff(g.value, true)

plt.plot(preds)
# plt.plot(losses)
plt.show()