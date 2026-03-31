import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 10, 11)
print(x)
y=x**2

y1 = np.exp(x)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x, y, label='y=x^2', color='blue', marker='o', linestyle='--', markersize=5, linewidth=2, markerfacecolor='red')
ax.plot(x, y1, label='y=exp(x)', color='green', marker='s', linestyle='-.', markersize=5, linewidth=2, markerfacecolor='orange')
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)
ax.legend(loc='upper left')

fig, ax = plt.subplots(1, 2, figsize=(5,2))
ax[0].plot(x, y)
ax[1].plot(x, np.exp(x))
plt.savefig("plot.png", dpi=200)
plt.show()