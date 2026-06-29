"""
General utility functions.
"""

import os
import numpy as np
import random
import torch

def numerical_stability_check(cov, device, epsilon=1e-6):
    """
    Check for numerical stability of covariance matrix.
    If not stable (i.e., not positive definite), add epsilon to diagonal.

    Parameters:
    cov (Tensor): The covariance matrix to check.
    epsilon (float, optional): The value to add to the diagonal if the matrix is not positive definite. Default is 1e-6.

    Returns:
    Tensor: The potentially adjusted covariance matrix.
    """
    num_added = 0
    if cov.dim() == 2:
        cov = (cov + cov.transpose(dim0=0, dim1=1)) / 2
    else:
        cov = (cov + cov.transpose(dim0=1, dim1=2)) / 2

    while True:
        try:
            # Attempt Cholesky decomposition; if it fails, the matrix is not positive definite
            torch.linalg.cholesky(cov)
            if num_added > 0.0001:
                print(
                    "Added {} to the diagonal of the covariance matrix.".format(
                        num_added
                    )
                )
            break
        except RuntimeError:
            # Add epsilon to the diagonal
            if cov.dim() == 2:
                cov = cov + epsilon * torch.eye(cov.size(0), device=device)
            else:
                cov = cov + epsilon * torch.eye(cov.size(1), device=device)
            num_added += epsilon
            epsilon *= 2
    return cov