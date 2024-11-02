from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(X, y, test_size=0.3, n_neighbors=5):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the classifier
    knn.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = knn.predict(X_test)
    
    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    # Load the Olivetti faces dataset
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    
    # Train and evaluate the model
    accuracy, report = train_and_evaluate(X, y, test_size=0.3, n_neighbors=5)
    
    # Print the results
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
