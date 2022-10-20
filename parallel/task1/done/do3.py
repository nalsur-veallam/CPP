import numpy as np

data = np.loadtxt("30S.out")

data[:,1] = data[:,1]**2*(data[:,1])**(-1)

np.savetxt('30S.out', data)
