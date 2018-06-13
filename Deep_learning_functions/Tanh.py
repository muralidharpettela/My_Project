"""
tanh
~~~~
Plots a graph of the tanh function."""

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, .1)
t = np.tanh(z)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, t)
ax.set_ylim([-1.0, 1.0])
ax.set_xlim([-5,5])
ax.grid(True, which='major')
#grid x-axis line color
a = ax.get_xgridlines()
b = a[3]
b.set_color('black')
b.set_linewidth(1.5)
#grid y-axis line color
c = ax.get_ygridlines()
d = c[4]
d.set_color('black')
d.set_linewidth(1.5)
#labels
ax.set_xlabel('z')
ax.set_title('tanh function')

plt.show()





