import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

df = pd.read_csv('./Admission.csv')

y_1 = pd.Categorical(df['Admission']).codes


x_n = ["GRE", "GPA"]

x_1 = df[x_n].values

x_1 = x_1 - x_1.mean()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta1 = pm.Normal('beta1', mu=0, sigma=2)
    beta2 = pm.Normal('beta2', mu=0, sigma=2)

    miu = alpha + beta1 * x_1[:,0] + beta2 * x_1[:,1]
    theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
    bd = pm.Deterministic('bd', -alpha/beta1 - beta2/beta1 * x_1[:,1])

    y1 = pm.Bernoulli('y1', p=theta, observed=y_1)

    trace = pm.sample(100)


idx = np.argsort(x_1[:,0])
bd = trace.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1])
plt.plot(x_1[:,0][idx], bd, color='k')
az.plot_hdi(x_1[:,0], trace.posterior['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

plt.savefig('figura.png')