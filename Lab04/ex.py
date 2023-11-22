# #nr clienti dustributie Poisson
# #lambda = 20 clienti/ora
# #timp plasare comanda distributie normala
# #E = 2 minute
# #sigma = 0.5 minute
# #preparare comanda distributie exponentiala
# #E = alpha minute

# #2) alpha maxim   timp < 15 minute   clienti care intra intr-o ora   prob = 95%

import numpy as np

def simulare_timp_asteptare(lambda_poisson,medie,sigma,alpha, nr_simulari):
    #lista cu timpii de asteptare
    timpi_asteptare = []
    for i in range(nr_simulari):
        #simulam numarul de clienti
        nr_clients = np.random.poisson(lambda_poisson)
        #simulam timpul necesar de plasare a unei comenzi
        timp_plasare = np.random.normal(medie, sigma, nr_clients)
        #simulam timpul necesar de preparare a comenzii
        timp_gatit = np.random.exponential(alpha, nr_clients)
        #timpul total + timpul mediu de asteptare
        total_time = np.sum(timp_plasare + timp_gatit)
        timp_mediu = total_time / nr_clients
        timpi_asteptare.append(timp_mediu)
    return np.array(timpi_asteptare)


def alpha_maxim(timp_maxim, probabilitate,lambda_poisson,medie,sigma,nr_simulari):
    #initializam max-alpha si probabilitatea sa cu 0
    max_alpha = 0
    max_alpha_prob = 0
    list_alpha = np.linspace(0.1, 15, 150)#punctele se alfa la o distanta de 0.1 

    for alpha in list_alpha:
        timp_asteptare = simulare_timp_asteptare(lambda_poisson,medie,sigma,alpha, nr_simulari)
        prob_aux = np.mean(timp_asteptare < timp_maxim)
        #verificam ca prob_aux sa satisfaca cerinta
        # (probabilitateava fi apropiata de 0.95, nu fix 0.95)
        if abs(prob_aux - probabilitate) < abs(max_alpha_prob - probabilitate):
            max_alpha_prob = prob_aux
            max_alpha = alpha
    return max_alpha

lambda_poisson = 20
medie = 2 
sigma = 0.5
nr_simulari = 1000

probabilitate = 0.95
timp_maxim = 15

alpha = alpha_maxim(timp_maxim,probabilitate,lambda_poisson,medie,sigma,nr_simulari)
print(f"Alpha maxim = {alpha:.1f}")

if alpha != 0:
    timp = simulare_timp_asteptare(lambda_poisson, medie, sigma, alpha,nr_simulari)
    print(f"Timp mediu de asteptare = {np.mean(timp):.2f} minute")