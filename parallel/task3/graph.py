import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


a = [1,2,4,8,16]#,24,32,40,48]

b1 = np.array([1.2261, 0.610249, 0.304222, 0.306491, 0.313792])#, 0.307258, 0.307673, 0.309571, 0.305841])
b2 = np.array([9.6927, 4.83328, 2.4904, 2.59897, 2.60672])#, 2.479, 2.47712, 2.44817, 2.461717])
b3 = np.array([33.9289, 20.8242, 18.3169, 9.03186, 9.05668])#, 9.02651, 10.3977, 17.7551, 17.9535])

gf1 = 0.25 / b1
gf2 = 2 / b2
gf3 = 6.75 / b3

#-O1
c1 = np.array([0.821796, 0.561367, 0.53388, 0.546619, 0.559571])#, 0.506114, 0.524108, 0.48937, 0.532601]
c2 = np.array([6.68657, 3.36021, 3.56908, 3.59022, 3.53954])#, 1.68272, 1.68048, 1.67723, 1.69055]
c3 = np.array([23.0622, 11.6881, 5.91584, 5.75698, 5.8112])#, 5.90023, 5.90897, 5.9073, 5.86346]

#-O0
d1 = np.array([1.11439, 0.564697, 0.329392, 0.321947, 0.323696])#, 0.324071, 0.325426, 0.321646, 0.32098]
d2 = np.array([9.78349, 5.03973, 2.77035, 2.74877, 2.70242])#, 2.69538, 2.69029, 2.69052, 2.69988]
d3 = np.array([35.7247, 27.4658, 20.8975, 9.35227, 10.2614])#, 9.69028, 12.3526, 13.1115, 14.7785]

#-O2
e1 = np.array([0.298521, 0.148911, 0.120331,  0.0763772, 0.0769012])#, 0.0786884, 0.0833252, 0.0850737, 0.07694])
e2 = np.array([2.76626, 1.53473, 0.717525, 0.773505, 0.711817])#, 0.697809, 0.74423, 0.78704, 0.663875])
e3 = np.array([10.1183, 5.00065, 2.6143, 2.43638, 2.32566])#, 2.30689, 2.44256, 2.29718, 2.30865])

#-O3
f1 = np.array([0.296921, 0.148138, 0.075873, 0.0749413, 0.0766497])#, 0.0783962, 0.0802681, 0.0765458, 0.0777073]
f2 = np.array([2.70575, 1.45369, 0.768605, 0.799776, 0.801098])#, 0.729007, 0.727285, 0.691764, 0.68866]
f3 = np.array([10.6196, 5.39373, 2.50486, 2.65179, 2.46381])#, 2.44041, 2.88861, 2.54018, 2.50178]

bf1 = 0.25 / e1
bf2 = 2 / e2
bf3 = 6.75 / e3

fig, ax = plt.subplots(2, 1, figsize = (12,16))


ax[0].plot(a, b1, '-', label = "N = 500")
ax[0].plot(a, b2, '-', label = "N = 1000")
ax[0].plot(a, b3, '-', label = "N = 1500")
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].set_xlabel("Num CPUs")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("Program execution time")
ax[0].legend()

ax[1].plot(a, gf1, '-', label = "N = 500")
ax[1].plot(a, gf2, '-', label = "N = 1000")
ax[1].plot(a, gf3, '-', label = "N = 1500")
ax[1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1].set_xlabel("Num CPUs")
ax[1].set_ylabel("GFLOPS")
ax[1].set_title("Performance")
ax[1].legend()

plt.show()
fig.savefig('first.pdf')

fig, ax = plt.subplots(2, 2, figsize = (16,16))


ax[0][0].plot(a, d1, '-', label = "N = 500")
ax[0][0].plot(a, d2, '-', label = "N = 1000")
ax[0][0].plot(a, d3, '-', label = "N = 1500")
ax[0][0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0][0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0][0].set_xlabel("Num CPUs")
ax[0][0].set_ylabel("Time (s)")
ax[0][0].set_title("Flag -О0")
ax[0][0].legend()

ax[0][1].plot(a, c1, '-', label = "N = 500")
ax[0][1].plot(a, c2, '-', label = "N = 1000")
ax[0][1].plot(a, c3, '-', label = "N = 1500")
ax[0][1].xaxis.set_minor_locator(AutoMinorLocator())
ax[0][1].yaxis.set_minor_locator(AutoMinorLocator())
ax[0][1].set_xlabel("Num CPUs")
ax[0][1].set_ylabel("Time (s)")
ax[0][1].set_title("Flag -О1")
ax[0][1].legend()

ax[1][0].plot(a, e1, '-', label = "N = 500")
ax[1][0].plot(a, e2, '-', label = "N = 1000")
ax[1][0].plot(a, e3, '-', label = "N = 1500")
ax[1][0].xaxis.set_minor_locator(AutoMinorLocator())
ax[1][0].yaxis.set_minor_locator(AutoMinorLocator())
ax[1][0].set_xlabel("Num CPUs")
ax[1][0].set_ylabel("Time (s)")
ax[1][0].set_title("Flag -О2")
ax[1][0].legend()

ax[1][1].plot(a, f1, '-', label = "N = 500")
ax[1][1].plot(a, f2, '-', label = "N = 1000")
ax[1][1].plot(a, f3, '-', label = "N = 1500")
ax[1][1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1][1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1][1].set_xlabel("Num CPUs")
ax[1][1].set_ylabel("Time (s)")
ax[1][1].set_title("Flag -О3")
ax[1][1].legend()

plt.show()
fig.savefig('Os.pdf')

fig, ax = plt.subplots(2, 1, figsize = (12,16))


ax[0].plot(a, e1, '-', label = "N = 500")
ax[0].plot(a, e2, '-', label = "N = 1000")
ax[0].plot(a, e3, '-', label = "N = 1500")
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].set_xlabel("Num CPUs")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("Program execution time")
ax[0].legend()

ax[1].plot(a, bf1, '-', label = "N = 500")
ax[1].plot(a, bf2, '-', label = "N = 1000")
ax[1].plot(a, bf3, '-', label = "N = 1500")
ax[1].xaxis.set_minor_locator(AutoMinorLocator())
ax[1].yaxis.set_minor_locator(AutoMinorLocator())
ax[1].set_xlabel("Num CPUs")
ax[1].set_ylabel("GFLOPS")
ax[1].set_title("Performance")
ax[1].legend()

plt.show()
fig.savefig('best.pdf')
