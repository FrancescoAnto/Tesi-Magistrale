import numpy as np
from solvers.utils.compute_coefficients import compute_coefficients

# --- Definizione della funzione base f(x) e sue derivate ---
def f(x):
    return 2.5

def df(x):
    return 0

def d2f(x):
    return 0


# --- Definition of the base flow ---
def base_flow(x, L, x1, x2, param = 1):
    r'''
    This code defines the base flow $u_B(x)$ as f(x) if x <= x1 or x >= x2, polynomial in between.
    The resulting function is C^2 continuous, with the coefficients of the polynomial determined by the conditions at x1 and x2.
    '''
    
    u = np.zeros_like(x, dtype=float)
    
    C = compute_coefficients(param, L, x1, x2)

    m1 = (x <= x1)
    m2 = (x > x1) & (x < x2)
    m3 = (x >= x2)
    
    u[m1] = f(x[m1])
    u[m2] = np.polyval(C, x[m2]) 
    u[m3] = f(x[m3])
    
    return u
