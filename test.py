import numpy as np
import matplotlib.pyplot as plt
from solvers.physics.get_base_flow import base_flow
from solvers.physics.get_initial_condition import initial_condition
from solvers.physics.get_boundary_condition import bc_sx

L = 1
x1 = 2*L
x2 = 3*L

x = np.linspace(0, 4, 1000)
t = np.linspace(0, 1, 100)
u = base_flow(x, L, x1, x2, param=5)
dudx = np.gradient(u, x)
u0 = initial_condition(x, L)
u_sx = bc_sx(t)

plt.figure(1)
plt.plot(x, u)
plt.grid()
plt.axis('equal')

plt.figure(2)
plt.plot(x, dudx)
plt.grid()
plt.axis('equal')

plt.figure(3)
plt.plot(x, u0)
plt.grid()
plt.axis('equal')

plt.figure(4)
plt.plot(t, u_sx)
plt.grid()
plt.axis('equal')

plt.show()



        
        
    

