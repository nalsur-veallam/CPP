import numpy as np
import pylab as plt

n = 2000
taum = 2.4e-9
taua = 2.1e-9
p = np.arange(1, 17, 1)
L = 6.5e-7
B=1.03e10
s=8

S = np.ones(16)*n**2*(n*taum + n*taua - taua)/(np.ones(16)/p*n**2*(n*taum + n*taua - taua) + (p - np.ones(16))*(np.ones(16)*L + n**2/B*s*np.ones(16)/p))

E = S/p


plt.plot(p, E, c='r', lw=3, label="E")
plt.plot(p, S, c='g', lw=3, label="S")
plt.legend()
plt.grid()
plt.show()

                                              
                                              
