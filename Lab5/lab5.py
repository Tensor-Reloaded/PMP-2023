import pymc3 as pm
import pandas

csv_data = pandas.read_csv("trafic.csv")

masini = csv_data["nr. masini"].values
minute = csv_data["minut"].values



intervale = [(0*60, 3*60), (3*60, 4*60), (4*60, 12*60), (12*60, 15*60), (15*60, 20*60)]

with pm.Model() as model:
    lambd = pm.Normal("lambda", mu=0, sigma=10)
    
    traffic_total = pm.Poisson('traffic', mu=lambd, observed=csv_data)

    intervale_distr = list()

    intervale_distr.append(pm.Poisson(f'lambda_1', mu=lambd,
                                       observed=masini[(minute >= 0) & (minute < 3*60)]))
    
    intervale_distr.append(pm.Poisson(f'lambda_2', mu=lambd * 1.3,
                                       observed=masini[(minute >= 3*60) & (minute < 4*60)]))
    
    intervale_distr.append(pm.Poisson(f'lambda_3', mu=lambd * 0.4, 
                                      observed=masini[(minute >= 4*60) & (minute < 12*60)]))
    
    intervale_distr.append(pm.Poisson(f'lambda_4', mu=lambd * 1.5, 
                                      observed=masini[(minute >= 12*60) & (minute < 15*60)]))
    
    intervale_distr.append(pm.Poisson(f'lambda_5', mu=lambd * 0.5,
                                      observed=masini[(minute >= 15*60) & (minute < 20*60)]))

with model:
    trace = pm.sample(2000, tune=5000)

pm.plot_trace(trace)