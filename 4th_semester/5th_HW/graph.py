import json
import pylab as plt
import numpy as np

with open('time.json') as f:
    data = json.loads(f.read())
    
time = data["time"]
fig, ax = plt.subplots()

ax = plt.plot(np.linspace(1, 31, 30), time, 'o-')
plt.xlabel("num of threads")
plt.ylabel("time(milliseconds)")
fig.savefig('graph.png')
