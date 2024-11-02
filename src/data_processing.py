from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

def load_data(data_dir='data'):
    """
    Loads the Olivetti Faces dataset, saves it locally if not present,
    and returns the data and labels.
    
    Parameters:
    - data_dir (str): Directory where the dataset is cached.

    Returns:
    - X (ndarray): Feature matrix (scaled).
    - y (ndarray): Labels for each image.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Define path to save/load dataset
    file_path = os.path.join(data_dir, 'olivetti_faces.pkl')
    scaler_path = os.path.join(data_dir, 'scaler.pkl')

    # Load dataset if already downloaded, otherwise fetch and save
    if os.path.exists(file_path):
        print("Loading dataset from local file.")
        data = joblib.load(file_path)
    else:
        print("Downloading dataset...")
        data = fetch_olivetti_faces()
        joblib.dump(data, file_path)
        print(f"Dataset saved to {file_path}")
    
    # Retrieve features and labels
    X, y = data.data, data.target
    
    # Scale features and save the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)  # Save the scaler for future use
    print(f"Scaler saved to {scaler_path}")

    return X_scaled, y

if __name__ == "__main__":
    try:
        # Load and preprocess the data
        X, y = load_data()
        
        # Save or print shapes for verification
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")
