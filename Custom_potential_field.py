import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x = np.random.rand(2)*4.0-2.0
y = np.random.rand(2)*4.0-2.0
z = x*np.exp(-x**2-y**2)


x = np.random.rand(2)*4.0-2.0
y = np.random.rand(2)*4.0-2.0
z = np.ones(2)*0.5



ti = np.linspace(-10.0, 10.0, 100)
plt.scatter(x, y, 100, z, cmap=cm.jet)
plt.xlabel('x')
plt.ylabel('y')
xx, yy = np.meshgrid(ti, ti)
plt.show()




rbf = Rbf(x, y, z, epsilon=0.2, function='gaussian')
zz = rbf(xx, yy)
print(zz)
print(zz.shape)
plt.imshow(zz, cmap=cm.jet, origin='lower', extent=[-2, 2, -2, 2])

plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, zz, cmap='viridis')
plt.show()