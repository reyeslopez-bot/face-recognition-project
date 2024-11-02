from sklearn.decomposition import PCA

def apply_pca(X, n_components=100):
    # PCA with 100 components
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    
    # Transform the original high-dimensional data into reduced dimesions
    X_pca = pca.fit_transform(X)
    return X_pca