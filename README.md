# Fed-MVKM: Federated Multi-View K-Means Clustering with Rectified Gaussian Kernel

## Overview

This package implements a combination of two advanced clustering algorithms:

1. Federated Multi-View K-Means Clustering (Fed-MVKM)
2. Rectified Gaussian Kernel Multi-View K-Means Clustering (MVKM-ED)

The implementation provides a privacy-preserving distributed learning framework for multi-view clustering while leveraging the enhanced discriminative power of rectified Gaussian kernels.

### About Fed-MVKM

Fed-MVKM is a novel privacy-preserving distributed learning framework designed for multi-view clustering that:

- Enables collaborative learning across distributed clients
- Preserves data privacy during the learning process
- Effectively handles heterogeneous data distributions
- Achieves robust clustering performance
- Implements adaptive weight learning mechanisms

### Repository Structure

```
Fed-MVKM/
├── Fed-MVKM-py/        # Python implementation
│   ├── mvkm_ed/        # Core Python package
│   ├── examples/       # Tutorials and examples
│   └── tests/         # Unit tests
└── matlab/            # MATLAB implementation
    ├── src/           # Source code
    └── examples/      # Example scripts
```

## Key Features

- Privacy-preserving federated learning for multi-view data
- Automatic view importance weight learning
- Rectified Gaussian kernel for enhanced distance computation
- Efficient distributed computation
- Scalable implementation for IoT and edge devices
- Automatic parameter adaptation
- GPU acceleration support

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.6.0
- scikit-learn >= 0.24.0

## Installation

### PyPI Package Status 📦

[![PyPI version](https://img.shields.io/pypi/v/mvkm-ed.svg)](https://pypi.org/project/mvkm-ed/)
[![Python versions](https://img.shields.io/pypi/pyversions/mvkm-ed.svg)](https://pypi.org/project/mvkm-ed/)

This package is officially published and verified on the Python Package Index (PyPI). You can:

- View the package at: [https://pypi.org/project/mvkm-ed/](https://pypi.org/project/mvkm-ed/)
- Check release history at: [https://pypi.org/project/mvkm-ed/#history](https://pypi.org/project/mvkm-ed/#history)
- Download statistics: [https://pypistats.org/packages/mvkm-ed](https://pypistats.org/packages/mvkm-ed)

### Quick Install

```bash
pip install mvkm-ed
```

## Usage

### Basic Example

```python
import numpy as np
from mvkm_ed import MVKMED, MVKMEDParams

# Create sample data
X1 = np.random.randn(100, 10)  # First view
X2 = np.random.randn(100, 15)  # Second view
X = [X1, X2]

# Set parameters
params = MVKMEDParams(
    cluster_num=3,
    points_view=2,
    alpha=2.0,
    beta=0.1,
    max_iterations=100,
    convergence_threshold=1e-4
)

# Create and fit model
model = MVKMED(params)
model.fit(X)

# Get cluster assignments
cluster_labels = model.index
```

### Federated Learning Example

```python
from mvkm_ed import FedMVKMED, FedMVKMEDParams

# Create client data
client_data = {
    'client1': [np.random.randn(100, 10), np.random.randn(100, 15)],
    'client2': [np.random.randn(100, 10), np.random.randn(100, 15)]
}

# Set federated parameters
fed_params = FedMVKMEDParams(
    cluster_num=3,
    points_view=2,
    alpha=2.0,
    beta=0.1,
    gamma=0.04,  # Federation parameter
    privacy_level=0.8
)

# Create and fit federated model
fed_model = FedMVKMED(fed_params)
fed_model.fit(client_data)

# Get global clustering results
global_labels = fed_model.get_global_labels()
```

## Datasets

### DHA (Depth-included Human Action) Dataset

The DHA dataset is an RGB-D multi-modal dataset for human action recognition and retrieval. This dataset represents a practical application of our federated multi-view clustering approach in action recognition using both depth and RGB information.

#### Dataset Details
- **Actions**: 23 different action categories
- **Subjects**: 21 different subjects performing actions
- **Views**: Two complementary data views:
  - Depth data (6144-dimensional feature vectors)
  - RGB data (110-dimensional feature vectors)

For detailed information about the dataset, please refer to the paper: "Human action recognition and retrieval using sole depth information" ([View Paper](https://dl.acm.org/doi/10.1145/2393347.2396381))

#### Example with DHA Dataset

```python
from mvkm_ed import FedMVKMED, FedMVKMEDParams
from mvkm_ed.datasets import load_dha

# Load DHA dataset with multiple views (depth and RGB)
X_dha, y_true = load_dha()  # Returns depth (6144-d) and RGB (110-d) features

# Split data for federated setup across different locations
client_data = {
    'site1': [X_dha[0][:150], X_dha[1][:150]],  # First 150 samples
    'site2': [X_dha[0][150:300], X_dha[1][150:300]],  # Next 150 samples
    'site3': [X_dha[0][300:], X_dha[1][300:]]  # Remaining samples
}

# Configure federated learning
fed_params = FedMVKMEDParams(
    cluster_num=23,  # Number of action categories
    points_view=2,  # Depth and RGB views
    alpha=2.0,
    beta=0.1,
    gamma=0.05,
    privacy_level=0.9
)

# Train federated model
fed_model = FedMVKMED(fed_params)
fed_model.fit(client_data)

# Evaluate clustering results
results = fed_model.evaluate(metrics=['nmi', 'ari'])
print(f"NMI Score: {results['nmi']:.3f}")
print(f"ARI Score: {results['ari']:.3f}")
```

## Parameters

### Basic Parameters

- `cluster_num`: Number of clusters
- `points_view`: Number of data views
- `alpha`: Exponent parameter to control view weights
- `beta`: Distance control parameter
- `max_iterations`: Maximum number of iterations
- `convergence_threshold`: Convergence criterion threshold

### Federated Parameters

- `gamma`: Federation parameter for client model updating
- `privacy_level`: Level of privacy preservation (0-1)
- `communication_rounds`: Maximum number of federation rounds
- `client_tolerance`: Convergence tolerance for client updates

### Algorithm Stages

1. **Initialization Stage**:

   - Set up central server
   - Initialize client configurations
   - Distribute initial parameters
2. **Client Stage**:

   - Local model optimization
   - View weight adaptation
   - Privacy preservation
3. **Federation Stage**:

   - Global model aggregation
   - Parameter synchronization
   - Convergence check
4. **Finalization Stage**:

   - Model evaluation
   - Results aggregation
   - Performance metrics computation

## Citation

If you use this code in your research, please cite our papers:

```bibtex
@ARTICLE{10810504,
  author={Yang, Miin-Shen and Sinaga, Kristina P.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Federated Multi-View K-Means Clustering}, 
  year={2025},
  volume={47},
  number={4},
  pages={2446-2459},
  doi={10.1109/TPAMI.2024.3520708}
}

@article{sinaga2024rectified,
  title={Rectified Gaussian Kernel Multi-View K-Means Clustering},
  author={Sinaga, Kristina P. and others},
  journal={arXiv},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Kristina P. Sinaga
- Email: kristinasinaga41@gmail.com

## Acknowledgments

This work was supported by:

- The National Science and Technology Council, Taiwan (Grant Number: NSTC 112-2118-M-033-004)
- GitHub Copilot for enhancing development efficiency and code quality
- The open-source community for their invaluable tools and libraries

Special thanks to GitHub Copilot for making the implementation process more efficient and helping to transform theoretical concepts into production-ready code. Its assistance significantly contributed to the development of both MATLAB and Python implementations.
