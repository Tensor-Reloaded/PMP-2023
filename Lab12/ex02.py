import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error

N_values = [100, 1000, 10000]
num_simulations = 20  

pi_values = []
error_values = []

for N in N_values:
    errors_for_N = []
    for _ in range(num_simulations):
        pi, error = estimate_pi(N)
        errors_for_N.append(error)
    
    mean_error = np.mean(errors_for_N)
    std_dev_error = np.std(errors_for_N)
    
    pi_values.append(pi)
    error_values.append(mean_error)

plt.errorbar(N_values, error_values, yerr=std_dev_error, fmt='o-', label='Error')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Error (%)')
plt.legend()

plt.savefig('ex02.png')