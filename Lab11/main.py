import arviz as az
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [10, 5, 0]
std_devs = [3, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
datas = np.array(mix)
az.plot_kde(datas)

idatas = []
models = []


cluster_values = [2, 3, 4]
for cluster in cluster_values:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means',mu=np.linspace(datas.min(), datas.max(), cluster),
                                sigma=10, shape=cluster
                                )
        sd = pm.HalfNormal('sd', sigma=10)
        y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=mix)
        idata = pm.sample(10, target_accept=0.9, random_seed=789, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)
        
[pm.compute_log_likelihood(idatas[i], model=models[i]) for i in range(3)]
comp_waic = az.compare(dict(zip([str(c) for c in cluster_values], idatas)),
                       method='BB-pseudo-BMA', ic="waic", scale="deviance")

print(comp_waic)
az.plot_compare(comp_waic)
plt.show()

comp_loo = az.compare(dict(zip([str(c) for c in cluster_values], idatas)),
                      method='BB-pseudo-BMA', ic="loo", scale="deviance")

print(comp_loo)
az.plot_compare(comp_loo)
plt.show()