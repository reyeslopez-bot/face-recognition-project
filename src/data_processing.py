from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    """
    Loads the Olivetti Faces dataset and returns the data and labels.
    """
    data = fetch_olivetti_faces()
    X = data.data  # Shape (400, 4096), each row is a flattened image
    y = data.target  # Labels for each image (0-39 representing individuals)
    return X, y

def preprocess_data(X):
    """
    Preprocesses the data by scaling features.
    
    Parameters:
    - X (ndarray): Feature matrix with shape (n_samples, n_features)
    
    Returns:
    - X_scaled (ndarray): Scaled feature matrix
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_data()
    X_scaled = preprocess_data(X)
    
    # Save or print shapes for verification
    print(f"Data shape: {X_scaled.shape}")
    print(f"Labels shape: {y.shape}")
