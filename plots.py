import matplotlib.pyplot as plt
import numpy as np

from functions import F

# meshgrid
delta = 0.25
x = np.arange(0.01, 4.0, delta)
y = np.arange(0.01, 4.0, delta)
X, Y = np.meshgrid(x, y, indexing='ij')

h = 1.0
g = 9.81
Z = F(X/np.sqrt(h/g), Y/np.sqrt(h/g), sign='-')


# plot
fig, ax = plt.subplots(1, 1, figsize=(5,4))
CS = ax.contour(X, Y, Z.real, 20)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_xlabel(r'$2 \pi f_n\sqrt{h/g}$') # dimensionless first-order frequencies
ax.set_ylabel(r'$2 \pi f_m\sqrt{h/g}$') # dimensionless first-order frequencies
ax.grid()
ax.set_aspect('equal')
#plt.savefig('results.png', dpi=300)
plt.show()
plt.close()



