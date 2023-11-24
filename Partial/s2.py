import numpy as np
import pymc as pm
from scipy import stats
import arviz as az
import matplotlib.pyplot as plt

SIZE = 100

with pm.Model() as model:
    waiting_times = stats.norm.rvs(5, 0.5, size=SIZE)
    
    n = pm.Poisson(mu=10, name='n')

    x = pm.Normal("Estimare", mu=n, sigma=1, observed=waiting_times)

    
    trace = pm.sample(2000, tune=1000)

    az.plot_posterior(trace)

    plt.title(f'Estimare:{n}')


plt.show()
