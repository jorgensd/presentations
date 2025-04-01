import matplotlib.pyplot as plt
import numpy as np

epsilon = np.array([10 ** (-i) for i in range(1, 10)])


def gd(epsilon):
    return np.log(1 / epsilon)


def sgd(epsilon):
    return 1 / epsilon


plt.figure(figsize=(10, 6))
plt.plot(epsilon, gd(epsilon), label="GD", marker="o")
plt.plot(epsilon, sgd(epsilon), label="SGD", marker="s")
plt.plot(epsilon, 1e2 * gd(epsilon), label="1e2 GD", marker="^")
plt.plot(epsilon, 1e5 * gd(epsilon), label="1e5 GD", marker="x")
plt.title("Cost of SDG vs GD")
plt.xlabel("Epsilon")
plt.ylabel("Cost")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig("cost.png")
