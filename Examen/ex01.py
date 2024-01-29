import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# a)
data = pd.read_csv('Titanic.csv')

y = data["Survived"]

print(len(y[y==1]),len(y[y==0])) 
Index = np.random.choice(np.flatnonzero(y==0), size=len(y[y==0])-len(y[y==1]), replace=False)


data = data.drop(labels=Index)

x_1 = data["Pclass"].values
x_2 = data["Age"].values
y = data["Survived"]

data = data.dropna(subset=["Age"])
x_1 = data["Pclass"].values
x_2 = data["Age"].values
y = data["Survived"].values

x_1_mean = x_1.mean()
x_2_mean = x_2.mean()


x_1_std = x_1.std()
x_2_std = x_2.std()

x_1 = (x_1 - x_1_mean) / x_1_std
x_2 = (x_2 - x_2_mean) / x_2_std

X = np.column_stack((x_1,x_2))


# b)
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=1, shape = 2)
    X_shared = pm.MutableData('x_shared', X) 
    mu = pm.Deterministic('μ',alpha + pm.math.dot(X_shared, beta))
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    y_pred = pm.Bernoulli("y_pred", p=theta, observed=y)
    idata = pm.sample(2000, return_inferencedata = True)

# c)
az.plot_forest(idata,hdi_prob=0.95,var_names=['beta'])
plt.savefig("ex01 c)")
# din grafic reiese ca Age influenteaza mai tare

# d)
obs_std1 = [(2-x_1_mean)/x_1_std,(30-x_2_mean)/x_2_std]
ppc = pm.sample_posterior_predictive(idata, model=model,var_names=["theta"])
y_ppc = ppc.posterior_predictive['theta'].stack(sample=("chain", "draw")).values
az.plot_posterior(y_ppc,hdi_prob=0.9)
plt.savefig("ex01 d)")
plt.show()
