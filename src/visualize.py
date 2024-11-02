import matplotlib.pyplot as plt
from data_processing import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os

def show_predictions(X_test, y_test, y_pred, num_images=20, save_path=None):
    """
    Displays a specified number of test images along with the model's predictions and true labels.
    """
    num_images = min(num_images, len(X_test), len(y_test), len(y_pred))  # Ensure we don't exceed available data
    rows = num_images // 5 + (num_images % 5 > 0)  # Calculate rows needed for display

    fig, axes = plt.subplots(rows, 5, figsize=(10, rows * 2))
    axes = axes.flatten()  # Flatten in case of a single row

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(X_test[i].reshape(64, 64), cmap='gray')
        ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
        ax.axis('off')

    # Hide any extra subplots if num_images < len(axes)
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()

    # Save plot if save_path is specified with higher DPI
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)  # Save with higher resolution
        print(f"Overwriting and saving visualization to {save_path}")

    plt.show()

# The main function to load data, train model, and generate predictions
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    show_predictions(X_test, y_test, y_pred, num_images=20, save_path="results/predictions.png")

if __name__ == "__main__":
    main()