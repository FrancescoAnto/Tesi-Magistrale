import numpy as np

def bc_sx(t):
    
    t1 = t[-1]/4
    
    u = np.zeros_like(t, dtype=float)
    
    m1 =  (t <= t1)
    
    u[m1] = np.sin(2 * np.pi * t[m1] / (t1))
    
    return u
