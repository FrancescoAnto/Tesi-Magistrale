import numpy as np

def initial_condition(x, L):

    u = np.zeros_like(x, dtype=float)

    return u

'''
x1 = L

u = np.zeros_like(x, dtype=float)

m1 =  (x <= x1)

u[m1] = np.sin(2 * np.pi * x[m1] / (x1))
'''