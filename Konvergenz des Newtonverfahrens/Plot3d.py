from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

X = np.arange(-5, 5, 1)
Y = np.arange(-5, 5, 1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

ax.scatter(X, Y, Z, c='r', marker='o')

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt

def z_function(x,y):
    return (x+1)**4+(y-1)**4

ax=plt.axes(projection="3d")

x = np.linspace(-2, 0, 100)
y = np.linspace(0, 2, 100)

X, Y = np.meshgrid(x,y)
Z = z_function(X, Y)
ax.plot_surface(X,Y,Z)
plt.show()


#z = np.linspace(0,30,100)
#x = np.sin(z)
#y = np.cos(z)
"""