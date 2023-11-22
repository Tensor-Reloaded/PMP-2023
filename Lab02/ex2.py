#4 servere
# #distribuit Γ(4, 3) pe primul server, Γ(4, 2) pe cel de-al doilea, Γ(5, 2) pe cel de-al treilea, 
# #şi Γ(5, 3) pe cel de-al patrulea (în milisecunde).
# #latenta lambda=4(milisecunde^-1)
# #P(S1)=0.25 P(S2)=0.25 P(S3)=0.3

#estimati probab ca timpul de servire X sa fie mai mare decat 3
#realizati un grafic al densitatii distributiei lui X

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#datele problemei
# gamma1=(4,3)
# gamma2=(4,2)
# gamma3=(5,2)
# gamma4=(5,3)
gammas=[(4,3),(4,2),(5,2),(5,3)]
lambda1=4
# prob1=0.25
# prob2=0.25
# prob3=0.3
# prob4=1-prob1-prob2-prob3
probs=[0.25,0.25,0.3,0.2]

nr_valori=10000

valori=[]
for i, (alpha,scale) in enumerate(gammas):
    n_i=nr_valori*probs[i]
    timp_procesare=stats.gamma.rvs(alpha,scale=1/scale,size=int(n_i))
    latenta=stats.expon.rvs(scale=1/lambda1,size=int(n_i))
    t=timp_procesare+latenta
    valori.append(t)

valori=np.array(t)

probab = np.mean(t>3)
print(f"Probabilitatea ca timpul de servire mai mare de 3ms: {probab}")

#grafic
plt.hist(t,bins=100)
plt.show()