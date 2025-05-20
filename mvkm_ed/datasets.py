"""DHA Dataset utilities for Fed-MVKM experiments."""

import numpy as np
from pathlib import Path
import scipy.io as sio
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import torch
from mvkm_ed.utils import MVKMEDDataProcessor

def load_dha_dataset(data_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load and preprocess the DHA dataset.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing the .mat files
        
    Returns
    -------
    views : List[np.ndarray]
        List of preprocessed views (RGB and Depth)
    labels : np.ndarray
        Ground truth labels
    """
    # Load raw data
    rgb_data = sio.loadmat(data_dir / 'RGB_DHA.mat')['RGB_DHA']
    depth_data = sio.loadmat(data_dir / 'Depth_DHA.mat')['Depth_DHA']
    labels = sio.loadmat(data_dir / 'label_DHA.mat')['label_DHA'].ravel()
    
    # Preprocess views
    processor = MVKMEDDataProcessor()
    views = processor.preprocess_views(
        [rgb_data, depth_data],
        scale=True,
        normalize=True
    )
    
    return views, labels

def create_client_partitions(
    views: List[np.ndarray],
    labels: np.ndarray,
    n_clients: int = 2,
    balanced: bool = True,
    random_state: Optional[int] = None
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, np.ndarray]]:
    """
    Split data into client partitions for federated learning.
    
    Parameters
    ----------
    views : List[np.ndarray]
        List of view matrices
    labels : np.ndarray
        Ground truth labels
    n_clients : int
        Number of clients to create
    balanced : bool
        Whether to create balanced partitions
    random_state : Optional[int]
        Random seed for reproducibility
        
    Returns
    -------
    client_data : Dict[int, List[np.ndarray]]
        Dictionary mapping client IDs to their view data
    client_labels : Dict[int, np.ndarray]
        Dictionary mapping client IDs to their labels
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = views[0].shape[0]
    
    if balanced:
        # Create balanced partitions
        indices = np.random.permutation(n_samples)
        client_size = n_samples // n_clients
        client_sizes = [client_size] * (n_clients - 1)
        client_sizes.append(n_samples - sum(client_sizes))
    else:
        # Create unbalanced partitions
        ratios = np.random.dirichlet(np.ones(n_clients))
        client_sizes = (ratios * n_samples).astype(int)
        client_sizes[-1] = n_samples - client_sizes[:-1].sum()
        indices = np.random.permutation(n_samples)
    
    client_data = {}
    client_labels = {}
    start_idx = 0
    
    for i in range(n_clients):
        end_idx = start_idx + client_sizes[i]
        client_indices = indices[start_idx:end_idx]
        
        # Assign data to client
        client_data[i] = [view[client_indices] for view in views]
        client_labels[i] = labels[client_indices]
        
        start_idx = end_idx
    
    return client_data, client_labels

def to_torch_tensors(
    client_data: Dict[int, List[np.ndarray]],
    device: torch.device
) -> Dict[int, List[torch.Tensor]]:
    """
    Convert client data to PyTorch tensors.
    
    Parameters
    ----------
    client_data : Dict[int, List[np.ndarray]]
        Client data as numpy arrays
    device : torch.device
        Device to place tensors on
        
    Returns
    -------
    Dict[int, List[torch.Tensor]]
        Client data as PyTorch tensors
    """
    return {
        cid: [torch.from_numpy(view).float().to(device) for view in views]
        for cid, views in client_data.items()
    }

def analyze_client_distribution(
    client_labels: Dict[int, np.ndarray]
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, int]]:
    """
    Analyze the distribution of classes across clients.
    
    Parameters
    ----------
    client_labels : Dict[int, np.ndarray]
        Dictionary mapping client IDs to their labels
        
    Returns
    -------
    client_class_dist : Dict[int, Dict[int, int]]
        Distribution of classes for each client
    global_class_dist : Dict[int, int]
        Global distribution of classes
    """
    client_class_dist = {}
    global_class_dist = {}
    
    for client_id, labels in client_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        client_class_dist[client_id] = dict(zip(unique, counts))
        
        for label, count in zip(unique, counts):
            global_class_dist[label] = global_class_dist.get(label, 0) + count
            
    return client_class_dist, global_class_dist
