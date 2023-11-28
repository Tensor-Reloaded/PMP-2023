import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def read_data():
    file_path = 'Prices.csv'
    df = pd.read_csv(file_path)

    prices = df['Price']
    cpu_speed = df['Speed']
    hard_size = np.log(df['HardDrive'])

    return prices, cpu_speed, hard_size

def main():
    prices, cpu_speed, hard_size = read_data()

    with pm.Model() as model_regression:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        sigma = pm.HalfCauchy('sigma', beta=5)

        mu = alpha + beta1 * cpu_speed + beta2 * hard_size

        price_obs = pm.Normal('price_obs', mu=mu, sigma=sigma, observed=prices)
        idata = pm.sample(2000, tune=2000, return_inferencedata=True)

    
    hdi_beta1 = az.hdi(idata['posterior']['beta1'], hdi_prob=0.95)
    hdi_beta2 = az.hdi(idata['posterior']['beta2'], hdi_prob=0.95)

    print("95% HDI pentru beta1:", hdi_beta1) # 13.16, 17.72 => cpu_speed are impact asupra pretului pt care intervalul nu contine 0
    print("95% HDI pentru beta2:", hdi_beta2) # -0.1685, 3.754 => nu stim daca hard_size are impact pt ca intervalul contine 0
    
    mu_predicted = pm.Deterministic('mu_predicted', alpha + beta1 * 33 + beta2 * np.log(540))

    # with model_regression:    
    #     idata_predicted = pm.sample_posterior_predictive(idata)
    #     hdi_90_pred = az.hdi(idata_predicted['mu_predicted'], hdi_prob=0.90)
    #     print("90% HDI pentru un computer cu o frecvenţă de 33 MHz şi un hard disk de 540 MB.:", hdi_90_pred)
    

if __name__ == '__main__':
    main()