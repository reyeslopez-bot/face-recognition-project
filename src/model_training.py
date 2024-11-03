import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_processing import load_data
import os
import json
import numpy as np

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def logo_evaluation(X, y, groups, n_neighbors=5):
    """Perform Leave-One-Group-Out Cross-Validation."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    logo = LeaveOneGroupOut()
    accuracies = []

    print("Performing Leave-One-Group-Out Cross-Validation...")
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    print(f"LOGO Cross-Validated Mean Accuracy: {mean_accuracy:.2f}")
    return mean_accuracy

def save_results(accuracy, report, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'model_results.json'), 'w') as f:
        json.dump({'accuracy': accuracy, 'classification_report': report}, f)
    print(f"[Model Training] Results saved to {results_dir}")

def main():
    X, y = load_data()
    groups = np.repeat(np.arange(40), 10)  # Each person has 10 images

    # LOGO Cross-Validation
    logo_accuracy = logo_evaluation(X, y, groups, n_neighbors=5)
    
    # Train-Test Split Evaluation (for comparison)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    split_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nTrain-Test Split Accuracy: {split_accuracy:.2f}")
    print("\nClassification Report:\n", report)
    
    save_results(split_accuracy, report)

if __name__ == "__main__":
    main()
