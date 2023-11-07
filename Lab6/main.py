import pymc as pm
import arviz as az
import itertools
import matplotlib.pyplot as plt

for Y, theta in itertools.product([0, 5, 10], [0.2, 0.5]):
    with pm.Model() as model:
        n = pm.Poisson(mu=10, name='n')

        nr_clienti_obs = pm.Binomial(f'nr_clienti_obs Y:{Y} Theta:{theta}', n=n, p=theta, observed=Y)

        trace = pm.sample(2000, tune=1000, cores=4)

        az.plot_posterior(trace)

        plt.title(f'Y:{Y} Theta:{theta}')


plt.show()
