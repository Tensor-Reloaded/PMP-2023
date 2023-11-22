# n client
# Y = numarul de clienti care cumpara un anumit produs
#   = distribuit Binomial(n, theta), theta = probabilitatea ca un client sa cumpere produsul
# distributia a priori pentru n este Poisson(10)

import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

# datele din problema
y = [0, 5, 10]
theta = [0.2, 0.5]
lambda_poisson = 10

def tema_6():
    # modelarea problemei folosind pymc
    with pm.Model() as model:
        # distributia a priori pentru n
        n = pm.Poisson("n", mu = lambda_poisson)
        # trecem prin toate combinatiile posibile de Y si theta
        for i in y:
            for t in theta:
                # valorile observate pentru y = i si theta = t
                obs = pm.Binomial(f"obs_{i}_{t}", n = n, p = t, observed = i)
        # distributia a posteriori
        trace = pm.sample(2000, tune = 100)
    # vizualizarea rezltatelor folosind az.plot_posterior
    az.plot_posterior(trace, var_names = ["n"])
    plt.xlabel("valori posibile pentru n")
    plt.ylabel("densitatea lui n")
    plt.show()

if __name__ == "__main__":
    tema_6()