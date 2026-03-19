import numpy as np

def crea_griglia_nd(bounds, num_points=10):
    """
    Creates an N-dimensional grid of points based on the provided bounds and number of points per dimension.
    
    Parameters:
    bounds (N,2): list or numpy, bounds [min, max] for each dimension.
    num_points (int): number of points per dimension (default 10).
    
    Returns:
    X_grid (num_points^N, N): numpy array with all combinations, where each row is a point in the N-dimensional space.
    """
    # Transform the input bounds into a numpy array for easier manipulation.
    bounds = np.array(bounds)
    N = len(bounds)
    
    vecs = []
    for i in range(N):
        min_val, max_val = bounds[i]
        vecs.append(np.linspace(min_val, max_val, num_points))
    
    # Create the grid using np.meshgrid, which will give us N arrays corresponding to each dimension.
    grids = np.meshgrid(*vecs, indexing='ij')
    
    # Flatten the grids and stack them column-wise to get a 2D array where each row is a point in the N-dimensional space.
    X_grid = np.column_stack([g.ravel() for g in grids])
    
    return X_grid
