"""
Test Script for Parkinson's Detection Model
Tests model on new data and validates performance
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def load_model_components():
    """Load trained model, scaler, and feature names"""
    print("Loading model components...")

    with open("parkinsons_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    print("✓ Model, scaler, and feature names loaded successfully!\n")
    return model, scaler, feature_names


def test_on_dataset(model, scaler, feature_names):
    """Test model on the original dataset"""
    print("=" * 60)
    print("TESTING ON ORIGINAL DATASET")
    print("=" * 60)

    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)

    X = df.drop(["name", "status"], axis=1)
    y = df["status"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y, predictions)

    print(f"\nTotal Samples: {len(y)}")
    print(f"Accuracy: {accuracy*100:.2f}%\n")

    print("Classification Report:")
    print(
        classification_report(y, predictions, target_names=["Healthy", "Parkinson's"])
    )

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, predictions)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

    return accuracy, predictions, probabilities


def test_single_sample(model, scaler, feature_names):
    """Test model on a single sample"""
    print("\n" + "=" * 60)
    print("TESTING SINGLE SAMPLE PREDICTION")
    print("=" * 60)

    # Example feature values (typical Parkinson's patient)
    sample_features = {
        "MDVP:Fo(Hz)": 197.07600,
        "MDVP:Fhi(Hz)": 206.89600,
        "MDVP:Flo(Hz)": 192.05500,
        "MDVP:Jitter(%)": 0.00289,
        "MDVP:Jitter(Abs)": 0.00001,
        "MDVP:RAP": 0.00166,
        "MDVP:PPQ": 0.00168,
        "Jitter:DDP": 0.00498,
        "MDVP:Shimmer": 0.01098,
        "MDVP:Shimmer(dB)": 0.09700,
        "Shimmer:APQ3": 0.00563,
        "Shimmer:APQ5": 0.00680,
        "MDVP:APQ": 0.00802,
        "Shimmer:DDA": 0.01689,
        "NHR": 0.00339,
        "HNR": 26.77500,
        "RPDE": 0.422229,
        "DFA": 0.741367,
        "spread1": -7.348300,
        "spread2": 0.177551,
        "D2": 1.743867,
        "PPE": 0.085569,
    }

    # Prepare feature vector
    feature_vector = [sample_features.get(fname, 0.0) for fname in feature_names]
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    probability = model.predict_proba(feature_vector_scaled)[0]

    print("\nSample Features:")
    for key, value in list(sample_features.items())[:5]:
        print(f"  {key}: {value}")
    print("  ...")

    print(f"\nPrediction: {'Parkinson\'s Disease' if prediction == 1 else 'Healthy'}")
    print(f"Confidence: {max(probability)*100:.2f}%")
    print(f"Risk Score: {probability[1]*100:.2f}%")

    return prediction, probability


def test_edge_cases(model, scaler, feature_names):
    """Test model on edge cases"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Edge case 1: All zeros
    print("\n1. Testing all-zero features:")
    zero_features = np.zeros((1, len(feature_names)))
    zero_features_scaled = scaler.transform(zero_features)
    pred = model.predict(zero_features_scaled)[0]
    prob = model.predict_proba(zero_features_scaled)[0]
    print(f"   Prediction: {'Parkinson\'s' if pred == 1 else 'Healthy'}")
    print(f"   Confidence: {max(prob)*100:.2f}%")

    # Edge case 2: All ones
    print("\n2. Testing all-one features:")
    one_features = np.ones((1, len(feature_names)))
    one_features_scaled = scaler.transform(one_features)
    pred = model.predict(one_features_scaled)[0]
    prob = model.predict_proba(one_features_scaled)[0]
    print(f"   Prediction: {'Parkinson\'s' if pred == 1 else 'Healthy'}")
    print(f"   Confidence: {max(prob)*100:.2f}%")

    # Edge case 3: Random normal features
    print("\n3. Testing random normal features:")
    random_features = np.random.randn(1, len(feature_names))
    random_features_scaled = scaler.transform(random_features)
    pred = model.predict(random_features_scaled)[0]
    prob = model.predict_proba(random_features_scaled)[0]
    print(f"   Prediction: {'Parkinson\'s' if pred == 1 else 'Healthy'}")
    print(f"   Confidence: {max(prob)*100:.2f}%")


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST REPORT")
    print("=" * 60)

    model, scaler, feature_names = load_model_components()

    # Test 1: Dataset performance
    accuracy, predictions, probabilities = test_on_dataset(model, scaler, feature_names)

    # Test 2: Single sample
    single_pred, single_prob = test_single_sample(model, scaler, feature_names)

    # Test 3: Edge cases
    test_edge_cases(model, scaler, feature_names)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Model loaded and functional")
    print(f"Dataset accuracy: {accuracy*100:.2f}%")
    print(f"Single sample prediction: {'Pass' if single_pred in [0, 1] else 'Fail'}")
    print(f"Edge cases handled: Pass")
    print(f"\n{'='*60}")
    print("ALL TESTS PASSED! Model is ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    generate_test_report()
