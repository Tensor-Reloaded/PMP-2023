from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

wins_j0 = 0
wins_j1 = 0

tests = 10000

for i in range(tests):
    who_starts = stats.binom.rvs(1, 0.5) # alegem cine incepe cu prob 0.5 pt fiecare jucator

    if who_starts == 0:
        n = stats.binom.rvs(1, 0.5) # daca este ales J0 atunci probabilitate e uniforma pt stema

        m = 0
        for j in range(n+1):
            m = stats.binom.rvs(1, 2/3) # J1 are prob 2/3 pentru stema

        if n >= m:
            wins_j0 += 1
        else:
            wins_j1 += +1

    else:
        n = stats.binom.rvs(1, 2/3) # daca incepe J1 atunci are prob. 2/3 pt stema

        m = 0
        for j in range(n+1):
            m = stats.binom.rvs(1, 0.5)

        if n >= m:
            wins_j1 += 1
        else:
            wins_j0 += +1


print(f'J0 wins: {wins_j0/tests} and J1 wins: {wins_j1/tests}')
# J0 wins: 0.4189 and J1 wins: 0.5811


model = BayesianNetwork([('S', 'J0', 'J1')])

cpd_start = TabularCPD(variable='S', variable_card=2, 
                       values=[[0.5], [0.5]])

cpd_j0 = TabularCPD(variable='R0', variable_card=2, values=[
                       [0.5, 1/3],
                       [0.5, 2/3]
                       ],
                  evidence=['S'],
                  evidence_card=[2])

cpd_winner = TabularCPD(variable='J1', variable_card=2, 
                   values=[
                       [],
                       []
                       ],
                  evidence=['R0'],
                  evidence_card=[2])


model.add_cpds(cpd_start, cpd_j0, cpd_winner)

assert model.check_model()

infer = VariableElimination(model)
result = infer.query(variables=['S'], evidence={'J1': 1})
print(result)
