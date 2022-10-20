import numpy as np

data = np.loadtxt("700.out")

data_S = np.concatenate((data[:, 0].reshape(-1,1), data[:, 2].reshape(-1,1)), axis=1)
data_E = np.concatenate((data[:, 0].reshape(-1,1), data[:, 3].reshape(-1,1)), axis=1)

np.savetxt('700S.out', data_S)
np.savetxt('700E.out', data_E)
