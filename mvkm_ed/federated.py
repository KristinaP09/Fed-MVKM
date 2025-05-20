"""Federated Multi-View K-Means Clustering with Exponential Distance.

This implementation extends the MVKM-ED algorithm to support federated learning
across distributed clients while preserving data privacy.

Authors: Kristina P. Sinaga
Date: May 2024
Version: 1.0

Copyright (c) 2023-2024 Kristina P. Sinaga
Contact: kristinasinaga41@gmail.com
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import logging
from tqdm.auto import tqdm
from .core import MVKMED, MVKMEDConfig

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FedMVKMEDConfig(MVKMEDConfig):
    """Configuration for Federated MVKM-ED algorithm."""
    
    gamma: float  # Model update parameter
    max_iterations: int = 10  # Default max iterations for federated learning
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError("gamma must be between 0 and 1")

class ClientModel:
    """Client-side model for federated learning."""
    
    def __init__(self, config: FedMVKMEDConfig, view_dimensions: List[int]):
        self.config = config
        self.view_dimensions = view_dimensions
        self.A = None  # Cluster centers
        self.V = None  # View weights
        self.U = None  # Memberships
        self.beta = None  # Distance parameter
        
    def initialize(self, data: List[torch.Tensor], global_centers: List[torch.Tensor]):
        """Initialize client model parameters."""
        self.A = [center.clone() for center in global_centers]
        self.V = torch.ones(self.config.points_view, device=data[0].device)
        self.V /= self.config.points_view
        self.beta = self._compute_initial_beta(data, global_centers)
    
    def _compute_initial_beta(self, data: List[torch.Tensor], 
                            centers: List[torch.Tensor]) -> torch.Tensor:
        """Compute initial beta parameter for distance adaptation."""
        beta = []
        n_samples = data[0].size(0)
        
        for h, (view_data, view_centers) in enumerate(zip(data, centers)):
            distances = []
            for k in range(self.config.cluster_num):
                dist = torch.sum((view_data - view_centers[k])**2, dim=1)
                distances.append(dist)
            distances = torch.stack(distances)
            
            # Compute statistics for beta calculation
            mean_dist = torch.mean(distances)
            range_dist = torch.max(distances) - torch.min(distances)
            beta_h = mean_dist * range_dist * (self.config.cluster_num / n_samples)
            beta.append(10.0 * beta_h)  # Scale factor similar to MATLAB
            
        return torch.tensor(beta, device=data[0].device)
    
    def update_memberships(self, data: List[torch.Tensor], 
                         global_centers: List[torch.Tensor]) -> torch.Tensor:
        """Update cluster memberships for local data."""
        n_samples = data[0].size(0)
        membership_values = torch.zeros((n_samples, self.config.cluster_num),
                                     device=data[0].device)
        
        for k in range(self.config.cluster_num):
            view_distances = torch.zeros((n_samples, self.config.points_view),
                                      device=data[0].device)
            
            for h in range(self.config.points_view):
                dist = torch.sum((data[h] - global_centers[h][k])**2, dim=1)
                kernel_dist = torch.exp(-self.beta[h] * dist)
                rectified_dist = 1 - kernel_dist
                view_distances[:, h] = (self.V[h]**self.config.alpha) * rectified_dist
            
            membership_values[:, k] = torch.sum(view_distances, dim=1)
        
        # Convert to one-hot encoding
        assignments = torch.argmin(membership_values, dim=1)
        self.U = torch.zeros_like(membership_values)
        self.U.scatter_(1, assignments.unsqueeze(1), 1)
        
        return self.U
    
    def update_centers(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Update local cluster centers."""
        new_centers = []
        
        for h in range(self.config.points_view):
            centers = torch.zeros((self.config.cluster_num, data[h].size(1)),
                                device=data[h].device)
            
            for k in range(self.config.cluster_num):
                mask = self.U[:, k] > 0
                if torch.any(mask):
                    weighted_data = data[h][mask] * self.U[mask, k].unsqueeze(1)
                    centers[k] = torch.sum(weighted_data, dim=0)
                    centers[k] /= torch.sum(self.U[:, k])
            
            new_centers.append(centers)
        
        self.A = new_centers
        return new_centers
    
    def update_weights(self, data: List[torch.Tensor]) -> torch.Tensor:
        """Update view weights."""
        V_terms = torch.zeros(self.config.points_view, device=data[0].device)
        
        for h in range(self.config.points_view):
            view_cost = 0.0
            for k in range(self.config.cluster_num):
                mask = self.U[:, k] > 0
                if torch.any(mask):
                    dist = torch.sum((data[h][mask] - self.A[h][k])**2, dim=1)
                    kernel_dist = torch.exp(-self.beta[h] * dist)
                    view_cost += torch.sum(self.U[mask, k] * (1 - kernel_dist))
            
            V_terms[h] = (1.0/view_cost)**(1.0/(self.config.alpha-1))
        
        self.V = V_terms / torch.sum(V_terms)
        return self.V

class FedMVKMED:
    """
    Federated Multi-View K-Means Clustering with Exponential Distance
    
    Features:
    - Distributed learning across multiple clients
    - Privacy-preserving computation
    - Efficient model aggregation
    - GPU acceleration
    - Progress monitoring and logging
    """
    
    def __init__(self, config: FedMVKMEDConfig):
        self.config = config
        self.global_centers = None
        self.client_models = {}
        self.history = {
            "objective_values": [],
            "center_updates": [],
            "client_objectives": {}
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.random_state is not None:
            torch.manual_seed(config.random_state)
            
    def _initialize_global_centers(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Initialize global cluster centers."""
        from sklearn.cluster import kmeans_plusplus
        centers = []
        
        for view in data:
            view_np = view.cpu().numpy()
            init, _ = kmeans_plusplus(view_np, self.config.cluster_num,
                                    random_state=self.config.random_state)
            centers.append(torch.from_numpy(init).float().to(self.device))
            
        return centers
    
    def _aggregate_models(self, client_centers: Dict[int, List[torch.Tensor]], 
                         client_sizes: Dict[int, int]) -> List[torch.Tensor]:
        """Aggregate client models into global model."""
        total_samples = sum(client_sizes.values())
        new_centers = []
        
        for h in range(self.config.points_view):
            weighted_sum = torch.zeros_like(self.global_centers[h])
            
            for client_id, centers in client_centers.items():
                weight = client_sizes[client_id] / total_samples
                weighted_sum += centers[h] * weight
            
            # Apply learning rate
            new_centers.append(
                self.global_centers[h] - self.config.gamma * weighted_sum
            )
            
        return new_centers
    
    def fit(self, client_data: Dict[int, List[np.ndarray]]) -> 'FedMVKMED':
        """
        Train the federated model across multiple clients.
        
        Parameters
        ----------
        client_data : Dict[int, List[np.ndarray]]
            Dictionary mapping client IDs to their local datasets
            
        Returns
        -------
        self : FedMVKMED
            Fitted model instance
        """
        logger.info("Starting Federated MVKM-ED training...")
        
        # Convert all client data to torch tensors
        torch_client_data = {
            cid: [torch.from_numpy(view).float().to(self.device) 
                 for view in views]
            for cid, views in client_data.items()
        }
        
        # Initialize global centers using combined data
        all_data = [torch.cat([client[h] for client in torch_client_data.values()])
                   for h in range(self.config.points_view)]
        self.global_centers = self._initialize_global_centers(all_data)
        
        # Initialize client models
        client_sizes = {cid: data[0].size(0) for cid, data in torch_client_data.items()}
        for client_id, data in torch_client_data.items():
            view_dims = [view.size(1) for view in data]
            self.client_models[client_id] = ClientModel(self.config, view_dims)
            self.client_models[client_id].initialize(data, self.global_centers)
            self.history["client_objectives"][client_id] = []
        
        # Main federation loop
        pbar = tqdm(range(1, self.config.max_iterations + 1),
                   disable=not self.config.verbose)
        
        try:
            for iteration in pbar:
                # Client updates
                client_centers = {}
                total_obj = 0.0
                
                for client_id, client in self.client_models.items():
                    data = torch_client_data[client_id]
                    
                    # Local model updates
                    client.update_memberships(data, self.global_centers)
                    client.update_centers(data)
                    client.update_weights(data)
                    
                    client_centers[client_id] = client.A
                    
                    # Compute client objective
                    obj = self._compute_client_objective(client, data)
                    self.history["client_objectives"][client_id].append(obj)
                    total_obj += obj
                
                # Server aggregation
                self.global_centers = self._aggregate_models(client_centers, client_sizes)
                
                # Track progress
                self.history["objective_values"].append(total_obj)
                self.history["center_updates"].append(
                    [center.cpu().numpy() for center in self.global_centers]
                )
                
                # Update progress bar
                pbar.set_description(f"Total Objective: {total_obj:.6f}")
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        return self
    
    def _compute_client_objective(self, client: ClientModel, 
                                data: List[torch.Tensor]) -> float:
        """Compute objective function value for a client."""
        obj = 0.0
        for h in range(self.config.points_view):
            view_obj = 0.0
            for k in range(self.config.cluster_num):
                mask = client.U[:, k] > 0
                if torch.any(mask):
                    dist = torch.sum((data[h][mask] - client.A[h][k])**2, dim=1)
                    kernel_dist = torch.exp(-client.beta[h] * dist)
                    view_obj += torch.sum(client.U[mask, k] * (1 - kernel_dist))
            obj += (client.V[h]**self.config.alpha) * view_obj
        return obj.item()
    
    def predict(self, client_data: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """Predict cluster assignments for client data."""
        predictions = {}
        
        # Convert data to torch tensors
        torch_data = {
            cid: [torch.from_numpy(view).float().to(self.device) 
                 for view in views]
            for cid, views in client_data.items()
        }
        
        for client_id, data in torch_data.items():
            if client_id in self.client_models:
                client = self.client_models[client_id]
            else:
                # Create new client model for unseen client
                view_dims = [view.size(1) for view in data]
                client = ClientModel(self.config, view_dims)
                client.initialize(data, self.global_centers)
            
            # Get memberships and convert to cluster assignments
            U = client.update_memberships(data, self.global_centers)
            predictions[client_id] = torch.argmax(U, dim=1).cpu().numpy()
        
        return predictions
