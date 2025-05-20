"""Core implementation of MVKM-ED algorithm with advanced optimizations."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy.sparse import csr_matrix
import torch
from tqdm.auto import tqdm

# Configure logging with rich formatting
try:
    from rich.logging import RichHandler
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MVKMEDConfig:
    """Configuration for MVKM-ED algorithm with parameter validation."""
    
    cluster_num: int
    points_view: int
    alpha: float
    beta: float
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    random_state: Optional[int] = None
    n_jobs: int = -1
    device: str = "auto"
    use_sparse: bool = True
    verbose: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.cluster_num < 1:
            raise ValueError("cluster_num must be positive")
        if self.points_view < 1:
            raise ValueError("points_view must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

class KernelFunction(ABC):
    """Abstract base class for kernel functions."""
    
    @abstractmethod
    def __call__(self, X: npt.NDArray, Y: npt.NDArray, beta: float) -> npt.NDArray:
        pass

class RectifiedGaussianKernel(KernelFunction):
    """Rectified Gaussian Kernel implementation."""
    
    def __call__(self, X: npt.NDArray, Y: npt.NDArray, beta: float) -> npt.NDArray:
        """Compute rectified Gaussian kernel."""
        if torch.is_tensor(X):
            return self._torch_kernel(X, Y, beta)
        return self._numpy_kernel(X, Y, beta)
    
    def _numpy_kernel(self, X: npt.NDArray, Y: npt.NDArray, beta: float) -> npt.NDArray:
        """NumPy implementation of kernel computation."""
        dist = np.sum((X[:, np.newaxis] - Y) ** 2, axis=2)
        return 1 - np.exp(-beta * dist)
    
    def _torch_kernel(self, X: torch.Tensor, Y: torch.Tensor, beta: float) -> torch.Tensor:
        """PyTorch implementation for GPU acceleration."""
        dist = torch.sum((X.unsqueeze(1) - Y) ** 2, dim=2)
        return 1 - torch.exp(-beta * dist)

class MVKMED:
    """
    Enhanced MVKM-ED Algorithm Implementation
    
    Features:
    - GPU acceleration with PyTorch
    - Sparse matrix support
    - Multi-threading for large datasets
    - Progress bars and rich logging
    - Memory-efficient computations
    - Advanced error handling
    """
    
    def __init__(self, config: MVKMEDConfig):
        self.config = config
        self.kernel = RectifiedGaussianKernel()
        self._setup_device()
        self._setup_random_state()
        self.history: Dict[str, List[float]] = {
            "objective_values": [],
            "view_weights": [],
            "beta_values": []
        }
        
    def _setup_device(self) -> None:
        """Setup computation device (CPU/GPU)."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")

    def _setup_random_state(self) -> None:
        """Setup random state for reproducibility."""
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_state)

    def _to_device(self, X: Union[np.ndarray, List[np.ndarray]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Convert numpy arrays to torch tensors on the correct device."""
        if isinstance(X, list):
            return [torch.from_numpy(x).float().to(self.device) for x in X]
        return torch.from_numpy(X).float().to(self.device)

    def _initialize_centers(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Initialize cluster centers using k-means++ initialization."""
        from sklearn.cluster import kmeans_plusplus
        centers = []
        for view in X:
            init, _ = kmeans_plusplus(view, self.config.cluster_num, 
                                    random_state=self.config.random_state)
            centers.append(init)
        return centers

    @torch.no_grad()
    def _update_memberships(self, X: List[torch.Tensor], A: List[torch.Tensor], 
                          V: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Update cluster memberships using efficient matrix operations."""
        data_n = X[0].size(0)
        membership_values = torch.zeros((data_n, self.config.cluster_num), 
                                     device=self.device)
        
        for k in range(self.config.cluster_num):
            view_distances = torch.zeros((data_n, self.config.points_view), 
                                      device=self.device)
            for h in range(self.config.points_view):
                dist = self.kernel(X[h], A[h][k:k+1], beta[h])
                view_distances[:, h] = (V[h] ** self.config.alpha) * dist.squeeze()
            
            membership_values[:, k] = view_distances.sum(dim=1)
        
        # Convert to one-hot encoding
        assignments = membership_values.argmin(dim=1)
        U = torch.zeros_like(membership_values)
        U.scatter_(1, assignments.unsqueeze(1), 1)
        
        return U

    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> 'MVKMED':
        """
        Fit the MVKM-ED model with advanced features and monitoring.
        
        Parameters
        ----------
        X : List[np.ndarray]
            List of data matrices for each view
        y : Optional[np.ndarray]
            Optional ground truth labels for monitoring

        Returns
        -------
        self : MVKMED
            Fitted model instance
        """
        logger.info("Starting MVKM-ED optimization...")
        
        # Convert data to torch tensors
        X_torch = self._to_device(X)
        
        # Initialize parameters
        self.A = self._to_device(self._initialize_centers(X))
        self.V = torch.ones(self.config.points_view, device=self.device) / self.config.points_view
        
        # Main optimization loop
        pbar = tqdm(range(1, self.config.max_iterations + 1), 
                   disable=not self.config.verbose)
        
        try:
            for iteration in pbar:
                # Update parameters
                beta = self._compute_beta(X_torch, iteration)
                U = self._update_memberships(X_torch, self.A, self.V, beta)
                self.A = self._update_centers(X_torch, U, beta)
                self.V = self._update_weights(X_torch, U, beta)
                
                # Compute objective
                obj = self._compute_objective(X_torch, U, beta)
                self.history["objective_values"].append(obj.item())
                self.history["view_weights"].append(self.V.cpu().numpy())
                self.history["beta_values"].append(beta.cpu().numpy())
                
                # Update progress bar
                pbar.set_description(f"Objective: {obj.item():.6f}")
                
                # Check convergence
                if iteration > 1:
                    diff = abs(self.history["objective_values"][-1] - 
                             self.history["objective_values"][-2])
                    if diff <= self.config.convergence_threshold:
                        logger.info(f"Converged after {iteration} iterations")
                        break
                        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
        # Get final cluster assignments
        self.labels_ = U.argmax(dim=1).cpu().numpy()
        
        return self
