#C - cutremur
#I - incendiu
#A - alarma

#P(C) = 0.05%
#P(I)=1%    P(I|C)=3%
#P(A)=0.01% P(A|C)=2%   P(A|I)=95%  P(A|I,C)=98%

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

#Definim modelul Bayesian
model_depozit = BayesianNetwork([('Cutremur','Incendiu'),
                                 ('Cutremur','Alarma'),
                                 ('Incendiu','Alarma')])

#Definim CPD-urile variabilelor
cpd_cutremur = TabularCPD(variable='Cutremur',variable_card=2,values=[[0.9995],[0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu',variable_card=2,values=[[0.99,0.97],[0.01,0.03]],
                          evidence=['Cutremur'],evidence_card=[2])
cpd_alarma = TabularCPD(variable='Alarma',variable_card=2,values=[[0.9999,0.98,0.05,0.02],[0.0001,0.02,0.95,0.98]],
                        evidence=['Cutremur','Incendiu'],evidence_card=[2,2])

#Adaugam distributiile la model
model_depozit.add_cpds(cpd_cutremur,cpd_incendiu,cpd_alarma)

#Verificam modelul
assert model_depozit.check_model()

#Aflam probabilitatea sa fi avut loc un cutremur stiind ca alarma a fost declansata
infer = VariableElimination(model_depozit)
prob_cutremur_alarma = infer.query(variables=['Cutremur'],evidence={'Alarma':1})
print(prob_cutremur_alarma)

#Aflam probabilitatea sa fi avut loc un incendiu stiind ca alarma de incendiu nu a fost activata
#vom folosi acelasi infer
prob_incendiu_not_alarma = infer.query(variables=['Incendiu'],evidence={'Alarma':0})
print(prob_incendiu_not_alarma)

#Reprezentam grafic problema
pos = nx.circular_layout(model_depozit)
nx.draw(model_depozit, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()