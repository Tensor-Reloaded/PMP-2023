import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def calculate():
    N = 10000
    x, y = stats.geom.rvs(0.3, 0.5, size=(2, N))
    inside = x > y**2
    aproximation = inside.sum() / N
    outside = np.invert(inside)

    return x, y, inside, outside, aproximation


if __name__ == "__main__":
    # a)
    x, y, inside, outside, aproximation = calculate()

    plt.figure(figsize=(8,8))

    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')

    plt.plot(0, 0, label=f'Aproximation: {aproximation}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)

    plt.savefig("ex02 a)")
    plt.show()

    # b)
    k = 30
    results = []
    for i in range(k):
        _, _, _, _, aproximation = calculate()
        results.append(aproximation)

    results_np = np.array(results)
    mean = results_np.mean()
    std = results_np.std()

    print(f"Mean: {mean}, Std: {std}") # Mean: 0.2677633333333333, Std: 0.004947826683392299