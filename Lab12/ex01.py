import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def posterior_grid(grid_points, heads, tails, prior_type='uniform'):
    grid = np.linspace(0, 1, grid_points)
    if prior_type == 'uniform':
        prior = np.repeat(1, grid_points)
    elif prior_type == 'cond':
        prior = (grid <= 0.5).astype(int)
    elif prior_type == 'abs':
        prior = abs(grid - 0.5)

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 250
h = data.sum()
t = len(data) - h


grid, posterior_uniform = posterior_grid(points, h, t, 'uniform')
grid, posterior_cond = posterior_grid(points, h, t, 'cond')
grid, posterior_abs = posterior_grid(points, h, t, 'abs')

plt.figure(figsize=(12, 9))
plt.plot(grid, posterior_uniform, '-o', label='uniform prior', color='blue')
plt.plot(grid, posterior_cond, '-o', label='conditional prior', color='green')
plt.plot(grid, posterior_abs, '-o', label='absolute prior', color='red')

plt.savefig('ex01.png')



