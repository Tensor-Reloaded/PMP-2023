from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'A'), ('C', 'I'), ('I', 'A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])

cpd_i = TabularCPD(variable='I', variable_card=2, 
                   values=[
                       [0.99, 0.97],
                       [0.01, 0.03]
                       ],
                  evidence=['C'],
                  evidence_card=[2])

cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[
                       [0.9999, 0.05, 0.98, 0.02],
                       [0.0001, 0.95, 0.02, 0.98]
                       ],
                  evidence=['C', 'I'],
                  evidence_card=[2, 2])

model.add_cpds(cpd_c, cpd_a, cpd_i)


assert model.check_model()

infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)

result = infer.query(variables=['I'], evidence={'A': 0})
print(result)

