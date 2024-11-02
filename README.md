# Face Recognition Project

## Project Overview
This project implements a face recognition model using the Olivetti Faces dataset. The goal is to classify different individuals using machine learning techniques. We use Principal Component Analysis (PCA) for dimensionality reduction and k-Nearest Neighbors (k-NN) as the classifier.

## Dataset
- **Name**: Olivetti Faces Dataset
- **Source**: Scikit-Learn
- **Description**: A dataset of 400 grayscale images (64x64) of faces from 40 different individuals.

## Setup and Installation

1. **Clone this repository**:
   ```bash
   git clone [your repo link]
   cd face-recognition-project
   ```

2. **Create a virtual environment** (recommended for isolating dependencies):
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On **macOS and Linux**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```

4. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the project**:
   - To execute the full pipeline (data loading, PCA, training, and evaluation), run:
     ```bash
     python src/model_training.py
     ```

## Project Structure

```plaintext
face-recognition-project/
├── data/                  # Directory for storing data or loading files if needed
├── notebooks/             # Jupyter notebooks (optional)
│   └── Face_Recognition_with_Olivetti_Faces.ipynb
├── src/                   # Source code for the project
│   ├── data_processing.py # Script for loading and processing data
│   ├── pca_reduction.py   # Script for dimensionality reduction with PCA
│   ├── model_training.py  # Script for model training and evaluation
│   └── visualize.py       # Script for visualizing results
├── results/               # Directory to store results, plots, and model outputs
│   └── example_predictions.png
├── README.md              # Project overview and setup instructions
└── requirements.txt       # List of Python dependencies
```

## Results
- **Accuracy**: Achieved ~XX% accuracy with k-NN and PCA.
- Example predictions can be found in the `results/` directory.

## Deactivating the Virtual Environment
When you're finished with the project, deactivate the virtual environment by running:
```bash
deactivate
```