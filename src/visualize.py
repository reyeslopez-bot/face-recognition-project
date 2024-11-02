import matplotlib.pyplot as plt

def show_predictions(X_test, y_test, y_pred):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    
    # Loop over the first 10 test images and display prediction results
    for i, ax in enumerate(axes.flat):
        # Reshape the image back to 64x64 pixels
        ax.imshow(X_test[i].reshape(64, 64), cmap='gray')
        
        # Set the image title as the predicted and true labels
        ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
        ax.axis('off')
    plt.show()