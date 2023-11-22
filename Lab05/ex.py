# 5 schimbari de comportament in trafic
# 4 - 7 constant
# 7 - 8 crestere
# 8 - 16 descrestere
# 16 - 19 crestere
# 19 - 24 descrestere
import numpy as np
import pandas as pd
import pymc3 as pm

def main():
    data = pd.read_csv(r"C:\Users\enbysenpai\PMP-2023\Lab05\trafic.csv")
    data = pd.Series(data['nr. masini'])

    # momentele in care au loc modificari in trafic (in minute)
    switchpoints = [180, 240, 720, 900]

    with pm.Model() as model:
        # estimam lambda pentru fiecare dintre cele 5 intervale
        lambdas_estimat = pm.Exponential('lambdas',1,shape=5)
        # alegem dintre valorile lui lambda in functie de index si de switchpoints
        lambdas = pm.math.switch(data.index < switchpoints[0], lambdas_estimat[0],
                                 pm.math.switch(data.index < switchpoints[1], lambdas_estimat[1],
                                 pm.math.switch(data.index < switchpoints[2], lambdas_estimat[2],
                                 pm.math.switch(data.index < switchpoints[3], lambdas_estimat[3],
                                 lambdas_estimat[4]))))
        # "legam" datele obtinute de model
        val_trafic = pm.Poisson('val_trafic', lambdas, observed = data)
        # sampling folosind NUTS
        trace = pm.sample(600,tune=600,cores=2)
    # print("Sumar complet al rezultatelor: ")
    # print(pm.summary(trace,var_names=['lambdas']))
    # cautam indexurile pentru switchpoints
    # cautam cele mai mari valori medii deoarece acelea sunt mai probabile sa se intample
    switchpoint_indices = np.argwhere(trace['lambdas'].mean(0) == trace['lambdas'].mean(0).max())
    # cautam indexurile pentru lambdas
    lambdas_mean = trace['lambdas'].mean(0)

    # Afișează capetele și valorile corespunzătoare
    switchpoint_indices = np.argwhere(trace['lambdas'].mean(0) == trace['lambdas'].mean(0).max())
    lambdas_mean = trace['lambdas'].mean(0)

    # Afișează capetele și valorile corespunzătoare
    for i in range(len(switchpoint_indices)):
        index = i
        if index == 0:
            left_endpoint = 0
        else:
            left_endpoint = switchpoints[index - 1]

        right_endpoint = switchpoints[index]

        print(f"Interval {i+1} (de la minutul {left_endpoint} până la minutul {right_endpoint}):")
        print(f"Capăt stânga: {lambdas_mean[index-1]}")
        print(f"Capăt dreapta: {lambdas_mean[index]}")


if __name__ == '__main__':
    main()