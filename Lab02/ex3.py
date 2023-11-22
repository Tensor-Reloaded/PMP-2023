#aruncarea de 10 ori a doua monezi
#prob moneda 2 sa obtinem stema = 0.3


#100 rezultate independente
#det grafic distributiile var aleatoare care numara rez posibile in cele 10 aruncari


import numpy as np
import matplotlib.pyplot as plt


#date problema
prob_moneda_normal=0.5
prob_moneda_masluit=0.3
nr_aruncari=10
nr_experimente=100


valori = {'ss':[],'sb':[],'bs':[],'bb':[]}


for i in range(nr_experimente):
    #simulam aruncarile in functie de probabilitati
    moneda_normala=np.random.choice(['s','b'],p=[prob_moneda_normal,1-prob_moneda_normal],size=nr_aruncari)
    moneda_masluita=np.random.choice(['s','b'],p=[prob_moneda_masluit,1-prob_moneda_masluit],size=nr_aruncari)
    #combinam rezultatele
    aruncari=list(zip(moneda_normala,moneda_masluita))
    #adaugam nuamrul de rezultate la vectorul de valori
    valori['ss'].append(aruncari.count(('s','s')))
    valori['sb'].append(aruncari.count(('s','b')))
    valori['bs'].append(aruncari.count(('b','s')))
    valori['bb'].append(aruncari.count(('b','b')))


#reprezentam grafic


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()


for i, (combination, counts) in enumerate(valori.items()):
    axes[i].hist(counts, bins=10)


plt.tight_layout()
plt.show()