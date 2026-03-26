import numpy as np

def step_bayesian_optimization(D, X_grid, l=1, sigma_f=1, sigma_n=1e-6):
    """
    Performs one step of Bayesian Optimization using Gaussian Process regression.
    
    Parameters:
    D (:,:) : The observed data points and their corresponding function values where each row is in the format [x (N,1), y (1,1)].
    X_grid (num_points^N, N): The grid of candidate points for optimization.
    l (float): The length scale parameter for the covariance function.
    sigma_f (float): The signal standard deviation parameter for the covariance function.
    sigma_n (float): The noise standard deviation parameter for the covariance function.
    
    Returns:
    """
    
    D = np.array(D)
    
    # Separate the input data into features (X) and target values (y).
    X = D[:, :-1]
    y = D[:, -1]


def rbf_kernel(X1, X2, l, sigma_f):
    # Computes the RBF kernel between two sets of points X1 and X2.
    
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1)
    cross_term = 2 * np.dot(X1, X2.T)
    
    # Applies the formula: ||x||^2 + ||y||^2 - 2x^Ty
    sq_dist = X1_sq + X2_sq - cross_term
    
    # Pevents negative values due to numerical issues, ensuring the distance is non-negative.
    sq_dist = np.maximum(sq_dist, 0.0)
    
    # Applies the RBF kernel formula: K(x, y) = sigma_f^2 * exp(-0.5 * ||x - y||^2 / l^2)
    K = (sigma_f ** 2) * np.exp(-0.5 * sq_dist / (l ** 2))
    
    return K
    
    
