import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


a = [1,2,4,8,16]

# 10000
t1 = 0.89
ts1 = np.array([1.715, 0.867, 0.448, 0.228, 0.220])
a1 = t1 / ts1
e1 = a1 / a

# 20000
t2 = 3.74
ts2 = np.array([6.544, 3.623, 2.087, 1.447, 2.236])
a2 = t2 / ts2
e2 = a2 / a

# 30000
t3 = 8.75
ts3 = np.array([16.002, 8.311, 4.244, 2.521, 4.468])
a3 = t3 / ts3
e3 = a3 / a

fig, ax = plt.subplots(2, 1, figsize = (12,16))


ax[0].plot(a, e1, '-', label = "N = 1e4")
ax[0].plot(a, e2, '-', label = "N = 2e4")
ax[0].plot(a, e3, '-', label = "N = 3e4")
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].set_xlabel("Num CPUs")
ax[0].set_ylabel("Efficiency")
ax[0].set_title("Efficiency")
ax[0].legend()

ax[1].plot(a, a1, '-', label = "N = 1e4")
ax[1].plot(a, a2, '-', label = "N = 2e4")
ax[1].plot(a, a3, '-', label = "N = 3e4")
ax[1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1].set_xlabel("Num CPUs")
ax[1].set_ylabel("Acceleration")
ax[1].set_title("Acceleration")
ax[1].legend()

plt.show()
fig.savefig('graph.pdf')
