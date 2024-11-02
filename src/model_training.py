import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from data_processing import load_data, preprocess_data


# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def train_and_evaluate(X, y, test_size=0.3, n_neighbors=5, use_cross_val=False, cv=5):
    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    if use_cross_val:
        # Perform cross-validation
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        mean_accuracy = scores.mean()
        return mean_accuracy, None  # No classification report for cross-validation
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the classifier
        knn.fit(X_train, y_train)
        
        # Predict on the test data
        y_pred = knn.predict(X_test)
        
        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        return accuracy, report

def main():
    # Load the dataset
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    
    # Option 1: Cross-validation
    mean_accuracy, _ = train_and_evaluate(X, y, n_neighbors=5, use_cross_val=True, cv=5)
    print(f"Cross-validated Mean Accuracy: {mean_accuracy:.2f}")
    
    # Option 2: Single train-test split
    accuracy, report = train_and_evaluate(X, y, test_size=0.3, n_neighbors=5, use_cross_val=False)
    print(f"\nTrain-Test Split Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()