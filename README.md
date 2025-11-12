# AI for Early Detection of Parkinson's from Voice

## Project Overview

This project implements a Minimum Viable Product (MVP) capable of predicting the risk of Parkinson's Disease (PD) using acoustic features extracted from a user's voice recording. It addresses the challenge by providing an end-to-end solution: a user interface for recording, a robust Flask API for feature extraction, and a highly accurate ensemble machine learning model for prediction.

The final web application gives a real-time risk assessment, along with key acoustic metrics derived from the recording, like fundamental frequency (Pitch), voice stability (Jitter), and Harmonics-to-Noise Ratio (HNR).

## Core Features (MVP Requirements)

1. **Voice Recording:** Frontend [(index.html)](./index.html) captures audio via the user's microphone.
2. **Audio Preprocessing & Feature Extraction:** The Flask API [(app.py)](./app.py) uses librosa to resample and extract 21 acoustic features compatible with the UCI Parkinson's dataset schema (e.g., pitch, jitter, shimmer, HNR, MFCCs).
3. **Real-Time Prediction:** A trained ensemble model processes the features and returns a risk prediction (Healthy/Parkinson's) and confidence score.
4. **Result Display:** The frontend clearly displays the risk label, confidence score, and specific acoustic metrics to the user.

## Technology Stack

- **Frontend:** HTML, CSS, JavaScript (Web Audio API for recording, Fetch API for communication).
- **Backend API:** Python, Flask, Flask-CORS.
- **Machine Learning:** Python, Scikit-learn (Random Forest, Gradient Boosting, SVM), Pandas, NumPy.
- **Audio Processing:** Librosa.
- **Data:** UCI Parkinson's Dataset.

## Setup and Run Instructions

**Prerequisites**
```
1. Python 3.x
2. All libraries listed in requirements.txt
```

## 1. Environment Setup

```
# Clone the repository
git clone [YOUR_REPO_LINK]
cd [YOUR_REPO_NAME]

# Install dependencies
pip install -r requirements.txt
```

## 2. Model Training

The [train_model.py](./train_model.py) script downloads the dataset, trains the ensemble model, performs evaluation, and saves the necessary files.
```
python train_model.py
```

This generates three files required by the API:
- [parkinsons_model.pkl](./parkinsons_model.pkl) (The trained VotingClassifier)
- [scaler.pkl](./scaler.pkl) (The fitted StandardScaler)
- [feature_names.pkl](./feature_names.pkl) (List of expected feature names)

## 3. Start the API Server

The [app.py](./app.py) file runs the Flask REST API.
```
python app.py
```

The server will start at http://localhost:5000

## 4. Run the Web Application

Open the [index.html](./index.html) file in your web browser to access the user interface. Ensure the API server is running before attempting a prediction.

# Model Performance Summary

The model uses a Voting Classifier (Ensemble) approach, combining the predictive power of Random Forest, Gradient Boosting, and a calibrated Support Vector Machine (SVM).

| Metric              | Score (on Test Set) |
|---------------------|---------------------|
| Accuracy            | 97.44%              |
| Precision (PD)      | 96.00%              |
| Recall (PD)         | 97.44%              |
| F1-Score (PD)       | 96.72%              |


This high performance demonstrates robustness and strong generalization capability on the provided dataset. The [train_model.py](./train_model.py) script also includes a 5-fold cross-validation, achieving a mean CV accuracy of ~92%.

## Primary Developer Contribution
The complete Artificial Intelligence component for the voice-based Parkinson's detection tool was developed solely by:
* **Khushi N** (https://github.com/khushi-n-murthy)

The repository was hosted by the Dev Keerthi P, to meet the hackathon submission requirement.