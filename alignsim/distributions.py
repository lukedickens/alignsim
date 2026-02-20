import numpy as np
from scipy.stats import beta

def get_beta_dist(mu, var, fragile=True, rtol=1e-3):
    """
    Creates a scipy.stats beta distribution from a mean and variance.
    """
    # Safety check for variance
    if var >= mu * (1 - mu):
        if fragile:
            raise ValueError(f"Variance {var} is too high for mean {mu}. "
                             f"Must be less than {mu * (1 - mu):.4f}")
        else:
            var = (1-rtol)*mu * (1 - mu)
    # Calculate alpha and beta
    term = (mu * (1 - mu) / var) - 1
    alpha_param = mu * term
    beta_param = (1 - mu) * term
    
    return beta(alpha_param, beta_param)

