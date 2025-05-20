"""Utility functions for MVKM-ED algorithm with advanced features."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pathlib import Path
import joblib
from dataclasses import asdict
import pandas as pd

class MVKMEDVisualizer:
    """Advanced visualization tools for MVKM-ED results."""
    
    def __init__(self, model: 'MVKMED'):
        self.model = model
        self.history = model.history
        plt.style.use('seaborn')
    
    def plot_convergence(self, figsize: tuple = (10, 6)) -> None:
        """Plot convergence trajectory with confidence intervals."""
        plt.figure(figsize=figsize)
        
        objectives = self.history['objective_values']
        x = np.arange(1, len(objectives) + 1)
        
        # Plot main convergence line
        plt.plot(x, objectives, 'b-', label='Objective Value')
        
        # Add rolling mean
        window = min(5, len(objectives))
        rolling_mean = pd.Series(objectives).rolling(window=window).mean()
        plt.plot(x, rolling_mean, 'r--', label=f'Rolling Mean (window={window})')
        
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('MVKM-ED Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key points
        min_obj = min(objectives)
        min_idx = objectives.index(min_obj)
        plt.annotate(f'Min: {min_obj:.4f}', 
                    xy=(min_idx + 1, min_obj),
                    xytext=(10, 10), textcoords='offset points')
        
        plt.tight_layout()
    
    def plot_view_weights(self, figsize: tuple = (12, 6)) -> None:
        """Plot evolution of view weights over iterations."""
        view_weights = np.array(self.history['view_weights'])
        
        plt.figure(figsize=figsize)
        for i in range(view_weights.shape[1]):
            plt.plot(view_weights[:, i], label=f'View {i+1}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Weight')
        plt.title('Evolution of View Weights')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add final weight annotations
        for i, final_weight in enumerate(view_weights[-1]):
            plt.annotate(f'V{i+1}: {final_weight:.3f}', 
                        xy=(len(view_weights) - 1, final_weight),
                        xytext=(10, 0), textcoords='offset points')
        
        plt.tight_layout()

class MVKMEDMetrics:
    """Advanced evaluation metrics for clustering quality."""
    
    @staticmethod
    def compute_metrics(X: List[np.ndarray], labels: np.ndarray, 
                       true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive clustering metrics."""
        metrics = {}
        
        # Internal metrics
        for i, view in enumerate(X):
            metrics[f'silhouette_view_{i+1}'] = silhouette_score(view, labels)
        
        metrics['silhouette_avg'] = np.mean([
            metrics[k] for k in metrics.keys() if 'silhouette_view_' in k
        ])
        
        # External metrics if true labels are provided
        if true_labels is not None:
            metrics['ari'] = adjusted_rand_score(true_labels, labels)
        
        return metrics

class MVKMEDDataProcessor:
    """Advanced data preprocessing for MVKM-ED."""
    
    @staticmethod
    def preprocess_views(views: List[np.ndarray], 
                        scale: bool = True,
                        normalize: bool = False) -> List[np.ndarray]:
        """Preprocess multiple views with advanced options."""
        processed_views = []
        
        for view in views:
            # Handle missing values
            if np.isnan(view).any():
                view = pd.DataFrame(view).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale features
            if scale:
                scaler = StandardScaler()
                view = scaler.fit_transform(view)
            
            # Normalize samples
            if normalize:
                norms = np.linalg.norm(view, axis=1, keepdims=True)
                view = view / (norms + 1e-8)
            
            processed_views.append(view)
        
        return processed_views

class MVKMEDPersistence:
    """Model persistence and serialization utilities."""
    
    @staticmethod
    def save_model(model: 'MVKMED', path: str) -> None:
        """Save model with all its components."""
        save_dict = {
            'config': asdict(model.config),
            'history': model.history,
            'labels': model.labels_ if hasattr(model, 'labels_') else None,
            'A': [a.cpu().numpy() for a in model.A] if hasattr(model, 'A') else None,
            'V': model.V.cpu().numpy() if hasattr(model, 'V') else None
        }
        joblib.dump(save_dict, path)
    
    @staticmethod
    def load_model(path: str) -> 'MVKMED':
        """Load model with all its components."""
        from .core import MVKMED, MVKMEDConfig
        
        save_dict = joblib.load(path)
        config = MVKMEDConfig(**save_dict['config'])
        model = MVKMED(config)
        
        model.history = save_dict['history']
        if save_dict['labels'] is not None:
            model.labels_ = save_dict['labels']
        if save_dict['A'] is not None:
            model.A = [torch.from_numpy(a).to(model.device) for a in save_dict['A']]
        if save_dict['V'] is not None:
            model.V = torch.from_numpy(save_dict['V']).to(model.device)
        
        return model
