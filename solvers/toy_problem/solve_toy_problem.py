import numpy as np
from solvers.physics.get_initial_condition import initial_condition
from solvers.physics.get_boundary_condition import bc_sx
from solvers.physics.get_base_flow import base_flow
from pde import CartesianGrid, ScalarField, PDE, MemoryStorage

def solve_toy_problem(bounds_x, bound_t, L, nu, param, nx, x1, x2):

    # Definizione del dominio
    pdegrid = CartesianGrid([[bounds_x[0], bounds_x[1]]], nx, periodic=False)
    x = pdegrid.axes_coords[0]
    t_bc = np.linspace(bound_t[0], bound_t[1], 500)

    # Definizione condizione iniziale 
    u0 = initial_condition(x, L)
    pdeic = ScalarField(pdegrid, data=u0)
    
    # Definizione condizioni al contorno
    u_sx = bc_sx(t_bc)
    def my_bc_func(adjacent_value, dx, x, t):
            return float(np.interp(t, t_bc, u_sx))
        
    boundaries = [{'value_expression': my_bc_func}, {'derivative': 0}]
    
    # Definizione dell'equazione differenziale
    ub = base_flow(x, L, x1, x2, param)
    dub_dx = np.gradient(ub, x)
    
    f_field = ScalarField(pdegrid, data=ub)
    g_field = ScalarField(pdegrid, data=dub_dx)
    
    dudt = 'a * laplace(u) - f * d_dx(u) + g * u'
    
    equation = PDE({'u': dudt}, bc = boundaries, consts={'a': nu, 'f': f_field, 'g': g_field})
    
    storage = MemoryStorage()
    
    risultato_finale = equation.solve(pdeic, bound_t,
        dt=1e-4,
        adaptive=True,
        tracker=["progress", storage.tracker(0.05)]
    )
    
    return risultato_finale, storage
    