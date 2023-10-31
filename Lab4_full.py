import numpy as np

epochs = 100

max_time = 15

def calc(alpha):
    accumulated_accuracy = 0

    all_clients_mean_times = []

    for _ in range(epochs):
        clients_count = np.random.poisson(lam=20, size=1)[0]
        
        order_times = np.random.normal(loc=2, scale=0.5, size=clients_count)

        cooking_times = np.random.exponential(scale=alpha, size=clients_count)

        total_times = order_times + cooking_times

        accuracy_cases = 0

        for x in total_times:
            if x < max_time:
                accuracy_cases += 1

        accuracy = accuracy_cases / clients_count    

        accumulated_accuracy += accuracy

        all_clients_mean_times.append(np.sum(total_times)/clients_count)



    all_clients_mean_time_per_all_epochs = np.sum(all_clients_mean_times) / epochs

    mean_accuracy = accumulated_accuracy / epochs

    return all_clients_mean_time_per_all_epochs, mean_accuracy


alpha = 1

epsilon = 0.01


best_all_clients_mean_time_per_all_epochs = 0
best_accuracy = 0

while True:
    all_clients_mean_time_per_all_epochs, actual_accuracy = calc(alpha)
    if actual_accuracy > 0.95:
        alpha += epsilon
        best_all_clients_mean_time_per_all_epochs = all_clients_mean_time_per_all_epochs
        best_accuracy = actual_accuracy
    else:
        break


print(f"Alpha:{alpha} \nAccuracy: {best_accuracy} \nMean Time Wait: {best_all_clients_mean_time_per_all_epochs}")


# Alpha:4.149999999999955 
# Accuracy: 0.951221224893471 
# Mean Time Wait: 6.1703905934187615