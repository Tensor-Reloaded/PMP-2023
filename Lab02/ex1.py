#2 mecanici
#t_mecanic1 -> lambda1=4h^-1
#t_mecanic2 -> lambda2=6h^-1
#mecanic2 serveste de 1.5 ori mai multi clienti decat mecanic1
#P(mecanic1)=40%
#X timp servire pentru un client


#gen 10000 valori pentru X
#estimati media si deviatia standard a lui X
#realizati un grafic al densitatii destributiei lui X


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#date problema
lambda1=4
lambda2=6
prob_mecanic1=0.4
nr_valori=10000


prob_mecanic2=1-prob_mecanic1


#array cu valori extrase din generari
valori_extrase=[]


#realizam experimentul
for i in range(nr_valori):
    #alegem random un mecanic, pe baza probabilitatii de servire
    mecanic_ales = np.random.choice(['mecanic1','mecanic2'],p=(prob_mecanic1,prob_mecanic2))
    #in functie de mecanicul ales, calculam variabila random exponentiala
    if mecanic_ales=='mecanic1':
        aux=stats.expon.rvs(scale=1/lambda1)
    else:
        aux=stats.expon.rvs(scale=1/lambda2)
    valori_extrase.append(aux)


#calculam media si variatia standard
media = np.mean(valori_extrase)
stdev=np.std(valori_extrase)
print(f"Media = {media}")
print(f"Variatia standard = {stdev}")


#realizam graficul densitatii distributiei
x= np.linspace(0.01,0.99,100)
p=stats.expon.pdf(x,scale=1/lambda1)*prob_mecanic1 + stats.expon.pdf(x,scale=1/lambda2)*prob_mecanic2
plt.plot(x,p,'k',linewidth=2)
plt.title("Densitatea distributiei lui X")


plt.show()