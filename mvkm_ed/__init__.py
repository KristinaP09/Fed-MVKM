"""
MVKM-ED: Rectified Gaussian Kernel Multi-View K-Means Clustering Package

This package implements the MVKM-ED algorithm for multi-view clustering,
including both centralized and federated learning variants.
"""

from .core import MVKMED, MVKMEDConfig
from .federated import FedMVKMED, FedMVKMEDConfig

__version__ = '1.0.0'
__author__ = 'Kristina P. Sinaga'
__email__ = 'kristinasinaga41@gmail.com'

__all__ = [
    'MVKMED', 
    'MVKMEDConfig',
    'FedMVKMED',
    'FedMVKMEDConfig'
]
