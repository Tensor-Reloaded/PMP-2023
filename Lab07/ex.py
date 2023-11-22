# b) CP intrare, mpg iesire

import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import numpy as np

def function():
    # citim datele din fisier
    # in fisierul csv, datele care nu exista sunt semnalate cu '?'
    # pastram doar coloanele de interes
    df = pd.read_csv('C:\\Users\\enbysenpai\\PMP-2023\\Lab07\\auto-mpg.csv', na_values = '?', usecols=['mpg','horsepower'])

    # curatam datele
    df = df.dropna(subset=['mpg','horsepower'])

    # graficul dependentei dintre CP si mpg
    plt.scatter(df['horsepower'], df['mpg'])
    plt.xlabel('Cai putere (horsepower)')
    plt.ylabel('Mile pe Galon (mpg)')
    plt.title('Relatia dintre Cai putere (horsepower) si Mile pe Galon (mpg)')
    plt.show()

    # definim modelul
    with pm.Model() as model:
        # prior pentru alpha
        alpha = pm.Normal('alpha', mu=0, sd=10)
        # prior pentru beta
        beta = pm.Normal('beta', mu=0, sd=10)
        epsilon = pm.HalfCauchy('epsilon',5)
        # dreapta folosita
        mu = alpha + beta * df['horsepower']
        # probabilitatea datelor observate
        mpg = pm.Normal('mpg', mu=mu, sd=epsilon, observed=df['mpg']) 

    # determinarea dreptei de regresie
    with model:
        trace = pm.sample(100, tune=100)

    az.summary(trace)

    # graficul 
    plt.scatter(df['horsepower'], df['mpg'], label='Date observate')

    pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(40, 230, 100), color='red', alpha=0.2, label='Distribuția predictiva a posteriori',
                                  lm=lambda x, sample: sample['alpha'] + sample['beta'] * x)
    plt.title('Horsepower vs. Mile per galon')
    plt.xlabel('Horsepower')
    plt.ylabel('Mile per galon (mpg)')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    function()