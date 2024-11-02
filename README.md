# Face Recognition Project

## Project Overview
This project implements a face recognition model using the Olivetti Faces dataset. The goal is to classify different individuals using machine learning techniques. The project leverages Principal Component Analysis (PCA) for dimensionality reduction and k-Nearest Neighbors (k-NN) as the classifier.

## Dataset
- **Name**: Olivetti Faces Dataset
- **Source**: Scikit-Learn
- **Description**: A dataset of 400 grayscale images (64x64) of faces from 40 different individuals.

## Setup and Installation

### 1. Running with Docker

To streamline dependencies and ensure consistent environments, the project is Dockerized.

1. **Build the Docker Image**:
   ```bash
   docker build -t face-recognition-project .
   ```

2. **Create Volumes for Data and Results**:
   To store data and results outside of the container, create persistent volumes:
   ```bash
   docker volume create face_recognition_data
   docker volume create face_recognition_results
   ```

3. **Run the Docker Container**:
   This command will execute all the scripts (data processing, model training, and visualization) and output results:
   ```bash
   docker run --rm -it \
       -v face_recognition_data:/app/data \
       -v face_recognition_results:/app/results \
       face-recognition-project
   ```

4. **Access Jupyter Notebook** (optional):
   To explore the Jupyter notebook in the project:
   ```bash
   docker run --rm -it \
       -v face_recognition_data:/app/data \
       -v face_recognition_results:/app/results \
       -p 8888:8888 \
       face-recognition-project \
       jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
   ```
   Open the URL shown in the terminal with a token to access the notebook interface.

### 2. Running Locally

If you prefer running the project locally without Docker:

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

6. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/Face_Recognition_with_Olivetti_Faces.ipynb
   ```

## Project Structure

```plaintext
face-recognition-project/
├── data/                  # Directory for storing data
├── notebooks/             # Jupyter notebooks (optional)
│   └── Face_Recognition_with_Olivetti_Faces.ipynb
├── src/                   # Source code for the project
│   ├── data_processing.py # Script for loading and processing data
│   ├── model_training.py  # Script for model training and evaluation
│   └── visualize.py       # Script for visualizing results
├── results/               # Directory to store results, plots, and model outputs
│   └── example_predictions.png
├── README.md              # Project overview and setup instructions
└── requirements.txt       # List of Python dependencies
```

## Results
- **Accuracy**: Achieved ~85% cross-validated accuracy with k-NN and PCA.
- **Visualizations**: Predictions are saved in the `results/` directory as `predictions.png`.

## Deactivating the Virtual Environment
When you're finished with the project, deactivate the virtual environment by running:
```bash
deactivate
```