import numpy as np

alpha = 15

epochs = 100

total_mean_time_per_client = 0

for _ in range(epochs):
    clients_count = np.random.poisson(lam=20, size=1)[0]
    
    order_times = np.random.normal(loc=2, scale=0.5, size=clients_count)

    cooking_times = np.random.exponential(scale=alpha, size=clients_count)


    mean_time_per_client = (np.sum(order_times) + np.sum(cooking_times)) / clients_count

    total_mean_time_per_client += mean_time_per_client

total_mean_time_per_client /=  epochs

print(f"Media pe client: {total_mean_time_per_client}")


