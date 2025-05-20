"""
MVKM-ED: Rectified Gaussian Kernel Multi-View K-Means Clustering Algorithm

This implementation presents a multi-view clustering approach using rectified
Gaussian kernels for distance computation. The algorithm effectively handles
multiple views of data while automatically learning view importance weights.

Authors: Kristina P. Sinaga
Date: May 2024
Version: 1.0

Copyright (c) 2023-2024 Kristina P. Sinaga
Contact: kristinasinaga41@gmail.com

This work was supported by the National Science and Technology Council, 
Taiwan (Grant Number: NSTC 112-2118-M-033-004)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MVKMEDParams:
    """Parameters for MVKM-ED algorithm"""
    cluster_num: int
    points_view: int
    alpha: float
    beta: float
    max_iterations: int = 100
    convergence_threshold: float = 1e-4

class MVKMED:
    """
    Rectified Gaussian Kernel Multi-View K-Means Clustering

    Parameters
    ----------
    params : MVKMEDParams
        Algorithm parameters

    Attributes
    ----------
    A : List[np.ndarray]
        Cluster centers for each view
    V : np.ndarray
        View weights
    U : np.ndarray
        Cluster membership matrix
    index : np.ndarray
        Cluster assignments
    param_beta : np.ndarray
        Computed beta parameters
    """

    def __init__(self, params: MVKMEDParams):
        self.params = params
        self.A = None
        self.V = None
        self.U = None
        self.index = None
        self.param_beta = None
        self.objective_values = []

    def _initialize_centers(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Initialize cluster centers using random selection."""
        data_n = X[0].shape[0]
        initial = np.random.permutation(data_n)[:self.params.cluster_num]
        return [x[initial] for x in X]

    def _compute_beta(self, X: List[np.ndarray], time: int) -> np.ndarray:
        """Compute beta parameters for distance adaptation."""
        data_n = X[0].shape[0]
        c = self.params.cluster_num
        return np.array([
            abs(np.sum(np.mean(x, axis=0)) * c / (time * data_n))
            for x in X
        ])

    def _update_memberships(self, X: List[np.ndarray]) -> np.ndarray:
        """Update cluster membership matrix."""
        data_n = X[0].shape[0]
        membership_values = np.zeros((data_n, self.params.cluster_num))

        for k in range(self.params.cluster_num):
            view_distances = np.zeros((data_n, self.params.points_view))
            for h in range(self.params.points_view):
                # Calculate distances in feature space
                dist = np.sum((X[h] - self.A[h][k])**2, axis=1)
                kernel_dist = np.exp(-self.param_beta[h] * dist)
                rectified_dist = 1 - kernel_dist
                # Weight by view importance
                view_distances[:, h] = (self.V[h]**self.params.alpha) * rectified_dist
            membership_values[:, k] = np.sum(view_distances, axis=1)

        # Convert to one-hot encoding
        assignments = np.argmin(membership_values, axis=1)
        U = np.zeros((data_n, self.params.cluster_num))
        U[np.arange(data_n), assignments] = 1
        return U

    def _update_centers(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Update cluster centers."""
        new_A = []
        for h in range(self.params.points_view):
            centers = np.zeros((self.params.cluster_num, X[h].shape[1]))
            for k in range(self.params.cluster_num):
                numerator = np.zeros(X[h].shape[1])
                denominator = 0
                
                dist = np.sum((X[h] - self.A[h][k])**2, axis=1)
                kernel_val = np.exp(-self.param_beta[h] * dist)
                weighted_kernel = (self.V[h]**self.params.alpha) * kernel_val
                
                numerator = np.sum(weighted_kernel[:, None] * self.U[:, k][:, None] * X[h], axis=0)
                denominator = np.sum(weighted_kernel * self.U[:, k])
                
                centers[k] = numerator / denominator
            new_A.append(centers)
        return new_A

    def _update_weights(self, X: List[np.ndarray]) -> np.ndarray:
        """Update view weights."""
        V_terms = np.zeros(self.params.points_view)
        for h in range(self.params.points_view):
            view_cost = 0
            for k in range(self.params.cluster_num):
                mask = self.U[:, k] > 0
                if np.any(mask):
                    dist = np.sum((X[h][mask] - self.A[h][k])**2, axis=1)
                    kernel_dist = np.exp(-self.param_beta[h] * dist)
                    view_cost += np.sum(self.U[mask, k] * (1 - kernel_dist))
            V_terms[h] = (1/view_cost)**(1/(self.params.alpha-1))
        return V_terms / np.sum(V_terms)

    def _compute_objective(self, X: List[np.ndarray]) -> float:
        """Compute objective function value."""
        obj = 0
        for h in range(self.params.points_view):
            view_obj = 0
            for k in range(self.params.cluster_num):
                mask = self.U[:, k] > 0
                if np.any(mask):
                    dist = np.sum((X[h][mask] - self.A[h][k])**2, axis=1)
                    kernel_dist = np.exp(-self.param_beta[h] * dist)
                    view_obj += np.sum(self.U[mask, k] * (1 - kernel_dist))
            obj += (self.V[h]**self.params.alpha) * view_obj
        return obj

    def fit(self, X: List[np.ndarray]) -> 'MVKMED':
        """
        Fit the MVKM-ED model to the data.

        Parameters
        ----------
        X : List[np.ndarray]
            List of data matrices for each view

        Returns
        -------
        self : MVKMED
            Fitted model
        """
        logger.info("Starting MVKM-ED algorithm initialization...")
        
        # Initialize parameters
        self.A = self._initialize_centers(X)
        self.V = np.ones(self.params.points_view) / self.params.points_view
        
        for time in range(1, self.params.max_iterations + 1):
            logger.info(f"Iteration {time}")
            
            # Update parameters
            self.param_beta = self._compute_beta(X, time)
            self.U = self._update_memberships(X)
            self.A = self._update_centers(X)
            self.V = self._update_weights(X)
            
            # Compute objective
            obj = self._compute_objective(X)
            self.objective_values.append(obj)
            
            # Check convergence
            if time > 1:
                diff = abs(self.objective_values[-1] - self.objective_values[-2])
                logger.info(f"Objective = {obj:.6f}, Difference = {diff:.6f}")
                if diff <= self.params.convergence_threshold:
                    logger.info(f"Algorithm converged after {time} iterations")
                    break
                    
        # Get final cluster assignments
        self.index = np.argmax(self.U, axis=1)
        return self
