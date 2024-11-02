import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import os
import json

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def cross_val_evaluation(X, y, n_neighbors=5, cv_folds=5):
    """Evaluate the model using cross-validation."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_scores = cross_val_score(knn, X, y, cv=cv_folds)
    return cv_scores.mean()

def train_test_evaluation(X, y, test_size=0.3, n_neighbors=5):
    """Evaluate the model using a single train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

def save_results(accuracy, report, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'model_results.json'), 'w') as f:
        json.dump({'accuracy': accuracy, 'classification_report': report}, f)

def main():
    # Load the dataset
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    
    # Run cross-validation
    cv_mean_accuracy = cross_val_evaluation(X, y, n_neighbors=5, cv_folds=5)
    print(f"Cross-validated Mean Accuracy: {cv_mean_accuracy:.2f}")
    
    # Run train-test split evaluation
    split_accuracy, report = train_test_evaluation(X, y, test_size=0.3, n_neighbors=5)
    print(f"\nTrain-Test Split Accuracy: {split_accuracy:.2f}")
    print("\nClassification Report:\n", report)
    save_results(split_accuracy, report)

if __name__ == "__main__":
    main()
