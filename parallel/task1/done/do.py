import numpy as np

data = np.loadtxt("16_700_node.txt")

S = (data[:,1]/data[0,1])**(-1)

E = (S/data[:,0])

data = np.concatenate((data, S.reshape(-1,1), E.reshape(-1,1)), axis=1)

np.savetxt('700.out', data)
