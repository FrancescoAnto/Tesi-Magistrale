import numpy as np

# --- Risolutore dei coefficienti dipendenti dal parametro libero ---
def compute_coefficients(param, L, x1, x2):
    from solvers.physics.get_base_flow import f, df, d2f
    
    """
    Calcola i coefficienti del polinomio di 6° grado.
    Il parametro 'param' agisce come termine libero c6 (coefficiente di x^6).
    Il solver calcola i restanti [c5, c4, c3, c2, c1, c0].
    """
    
    # Matrice dei coefficienti delle derivate per i termini da x^5 a x^0
    M_cond = np.array([
        [1, 1, 1, 1, 1, 1],     # Condizione C^0
        [5, 4, 3, 2, 1, 0],     # Condizione C^1
        [20, 12, 6, 2, 0, 0],   # Condizione C^2
    ])
    
    def poly_matrix(x):
        poly = x ** np.arange(5, -1, -1) # [x^5, x^4, x^3, x^2, x^1, x^0]
        
        X = np.array([
            poly,
            poly / x,
            poly / (x**2)
        ])
        
        X[1, -1] = 0
        X[2, -2:] = 0
        
        return X

    X = np.vstack((poly_matrix(x1), poly_matrix(x2)))
    M = np.vstack((M_cond, M_cond))

    A = M * X
     
    # Sottraiamo f(x) - param * x^6 e le relative derivate (6x^5 e 30x^4)
    b_vec = np.array([
        f(x1) - param * (x1**6),
        df(x1) - 6 * param * (x1**5),
        d2f(x1) - 30 * param * (x1**4),
        f(x2) - param * (x2**6),
        df(x2) - 6 * param * (x2**5),
        d2f(x2) - 30 * param * (x2**4)
    ])
    
    # Risoluzione del sistema lineare per i termini da c5 a c0
    C_solved = np.linalg.solve(A, b_vec)
    
    # Il vettore C finale contiene 7 elementi: [c6, c5, c4, c3, c2, c1, c0]
    C = np.hstack((param, C_solved))

    return C