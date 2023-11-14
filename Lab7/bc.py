import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np

data = pd.read_csv('auto-mpg.csv')
df = data[['horsepower', 'mpg']]
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()


with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    y_est = alpha + beta * df['horsepower']

    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=df['mpg'])
    trace = pm.sample(1000, tune=1000)

az.plot_posterior(trace)


alpha_mean = np.mean(trace['alpha'])
beta_mean = np.mean(trace['beta'])

df['estimated_mpg'] = alpha_mean + beta_mean * df['horsepower']

plt.figure(figsize=(20, 10))
plt.scatter(df['horsepower'], df['mpg'], alpha=0.5)
plt.plot(df['horsepower'], df['estimated_mpg'], color='red', label='Regresie')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.savefig('bc.png')
