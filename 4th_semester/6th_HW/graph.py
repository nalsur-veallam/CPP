import pylab as plt
import numpy as np

time = np.loadtxt('forpy.txt')

fig, ax = plt.subplots(figsize=(15,12))

ax = plt.plot(np.linspace(1000, 100000, 100), time, 'o-')
plt.xlabel("num of tasks")
plt.ylabel("time(milliseconds)")
plt.grid()
plt.title('Program running time depending on the number of tasks with 8 threads')
fig.savefig('graph.pdf')
