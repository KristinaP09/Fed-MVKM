"""Example usage of Federated MVKM-ED algorithm on DHA dataset."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from mvkm_ed import FedMVKMED, FedMVKMEDConfig
from mvkm_ed.utils import MVKMEDDataProcessor, MVKMEDMetrics, MVKMEDVisualizer
import scipy.io as sio
import matplotlib.pyplot as plt

# Load DHA dataset
def load_dha_data():
    """Load and preprocess DHA dataset."""
    rgb_data = sio.loadmat('RGB_DHA.mat')['RGB_DHA']
    depth_data = sio.loadmat('Depth_DHA.mat')['Depth_DHA']
    labels = sio.loadmat('label_DHA.mat')['label_DHA'].ravel()
    
    # Normalize data
    processor = MVKMEDDataProcessor()
    views = processor.preprocess_views([rgb_data, depth_data])
    
    return views, labels

# Split data into clients
def create_client_data(views, n_clients=2):
    """Split data into client partitions."""
    n_samples = views[0].shape[0]
    indices = np.random.permutation(n_samples)
    client_size = n_samples // n_clients
    
    client_data = {}
    for i in range(n_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < n_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]
        
        client_data[i] = [view[client_indices] for view in views]
    
    return client_data

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading DHA dataset...")
    views, labels = load_dha_data()
    n_clusters = len(np.unique(labels))
    n_views = len(views)
    
    # Create federated setup
    client_data = create_client_data(views, n_clients=2)
    
    # Configure federated learning
    config = FedMVKMEDConfig(
        cluster_num=n_clusters,
        points_view=n_views,
        alpha=15.0,  # View weight control
        beta=1.0,   # Initial distance parameter
        gamma=0.04,  # Model update rate
        max_iterations=10,
        convergence_threshold=1e-4,
        random_state=42,
        verbose=True
    )
    
    # Initialize and train federated model
    print("\nInitializing federated training...")
    model = FedMVKMED(config)
    model.fit(client_data)
    
    # Get predictions for all clients
    predictions = model.predict(client_data)
    
    # Evaluate results
    print("\nEvaluating clustering results...")
    all_predictions = np.concatenate([pred for pred in predictions.values()])
    metrics = MVKMEDMetrics.compute_metrics(views, all_predictions, labels)
    
    print("\nClustering Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results
    visualizer = MVKMEDVisualizer(model)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    visualizer.plot_convergence()
    plt.title("Convergence Analysis")
    
    plt.subplot(1, 2, 2)
    visualizer.plot_view_weights()
    plt.title("View Weight Evolution")
    
    plt.tight_layout()
    plt.show()
