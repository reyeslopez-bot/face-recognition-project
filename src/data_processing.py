from sklearn.datasets import fetch_olivetti_faces

def load_data():
    # Fetch the Olivetti faces dataset
    faces = fetch_olivetti_faces()
    
    # X is the data matrix(each row represents an image as a flattened array)
    # y contains the labels corresponding to each image (class IDs, representing different individuals)
    X, y = faces.data, faces.target
    return X, y