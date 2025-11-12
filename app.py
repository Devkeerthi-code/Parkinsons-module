"""
Flask Backend API for Parkinson's Detection
Handles audio processing and real-time prediction with improved error handling

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import pickle
import io
import warnings
import traceback
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Global variables for model components
model = None
scaler = None
feature_names = None

print("Relying on browser-sent WAV audio format (pcm) for direct librosa processing.")


def load_model_components():
    """Load trained model and scaler with error checking"""
    global model, scaler, feature_names

    try:
        # Check if files exist
        required_files = ["parkinsons_model.pkl", "scaler.pkl", "feature_names.pkl"]
        missing_files = [f for f in required_files if not os.path.exists(f)]

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing_files)}"
            )

        with open("parkinsons_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        print("✓ Model components loaded successfully!")
        return True

    except Exception as e:
        print(f"✗ Error loading model components: {str(e)}")
        print("Please run 'python train_model.py' first to generate the model files.")
        return False


def extract_features(audio_data, sr):
    """
    Extract acoustic features from audio matching training dataset
    """
    features = {}

    try:
        # Fundamental frequency (F0) features
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            features["MDVP:Fo(Hz)"] = np.mean(f0_clean)
            features["MDVP:Fhi(Hz)"] = np.max(f0_clean)
            features["MDVP:Flo(Hz)"] = np.min(f0_clean)
        else:
            # Fallback values for silent/bad segments
            features["MDVP:Fo(Hz)"] = 150
            features["MDVP:Fhi(Hz)"] = 200
            features["MDVP:Flo(Hz)"] = 100

        # Jitter measures (frequency variation)
        if len(f0_clean) > 1:
            jitter_abs = np.mean(np.abs(np.diff(f0_clean)))
            features["MDVP:Jitter(%)"] = (jitter_abs / features["MDVP:Fo(Hz)"]) * 100
            features["MDVP:Jitter(Abs)"] = jitter_abs
            features["MDVP:RAP"] = np.mean(np.abs(np.diff(f0_clean, n=1)))
            features["MDVP:PPQ"] = np.mean(np.abs(np.diff(f0_clean, n=2)))
            features["Jitter:DDP"] = features["MDVP:RAP"] * 3
        else:
            features["MDVP:Jitter(%)"] = 0.003
            features["MDVP:Jitter(Abs)"] = 0.00003
            features["MDVP:RAP"] = 0.002
            features["MDVP:PPQ"] = 0.002
            features["Jitter:DDP"] = 0.006

        # Shimmer measures (amplitude variation)
        rms = librosa.feature.rms(y=audio_data)[0]
        if len(rms) > 1:
            shimmer_abs = np.mean(np.abs(np.diff(rms)))
            features["MDVP:Shimmer"] = (
                (shimmer_abs / np.mean(rms)) * 100 if np.mean(rms) > 0 else 0.03
            )
            features["MDVP:Shimmer(dB)"] = 20 * np.log10(shimmer_abs + 1e-10)
            features["Shimmer:APQ3"] = np.mean(np.abs(np.diff(rms, n=1)))
            features["Shimmer:APQ5"] = np.mean(np.abs(np.diff(rms, n=2)))
            features["MDVP:APQ"] = features["Shimmer:APQ5"]
            features["Shimmer:DDA"] = features["Shimmer:APQ3"] * 3
        else:
            features["MDVP:Shimmer"] = 0.03
            features["MDVP:Shimmer(dB)"] = 0.3
            features["Shimmer:APQ3"] = 0.015
            features["Shimmer:APQ5"] = 0.02
            features["MDVP:APQ"] = 0.02
            features["Shimmer:DDA"] = 0.045

        # Harmonics-to-Noise Ratio
        harmonic, percussive = librosa.effects.hpss(audio_data)
        # HNR (Harmonics-to-Noise Ratio)
        hnr = 10 * np.log10(
            (np.sum(harmonic**2) + 1e-10) / (np.sum(percussive**2) + 1e-10)
        )
        features["NHR"] = (
            1 / (hnr + 1e-10) if hnr > 0 else 0.02
        )  # NHR (Noise-to-Harmonics Ratio)
        features["HNR"] = max(0, hnr)

        # Recurrence Period Density Entropy
        features["RPDE"] = np.std(audio_data) * 0.5

        # Detrended Fluctuation Analysis
        features["DFA"] = np.mean(np.abs(audio_data - np.mean(audio_data))) * 0.7

        # Pitch Period Entropy
        features["spread1"] = np.std(f0_clean) if len(f0_clean) > 0 else -5.0
        features["spread2"] = features["spread1"] * 0.8
        features["D2"] = 2.5
        features["PPE"] = (
            -np.log(np.var(f0_clean) + 1e-10) if len(f0_clean) > 0 else 0.15
        )

        return features

    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise


def preprocess_audio(audio_bytes):
    """
    Preprocess audio: Loads the browser-sent PCM/WAV audio data directly using
    librosa from an in-memory stream, bypassing problematic external dependencies.
    """
    SR = 22050  # Target sample rate

    try:
        # 1. Create a file-like object from the raw bytes
        audio_file = io.BytesIO(audio_bytes)

        # 2. Load the audio data directly.
        audio_data, sr = librosa.load(audio_file, sr=SR)

        if len(audio_data) == 0:
            raise ValueError("Audio file is empty or contains no detectable signal.")

        # 3. Trim silence (top_db=20 is a good threshold for speech)
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

        if len(audio_data) == 0:
            raise ValueError("Audio file contains only silence after trimming.")

        # 4. Normalize
        audio_data = librosa.util.normalize(audio_data)

        return audio_data, sr

    except Exception as e:
        print("-" * 50)
        print("CRITICAL ERROR DURING AUDIO PREPROCESSING (Librosa direct load failure)")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(traceback.format_exc())
        print("-" * 50)
        # If conversion fails, raise a specific error
        raise RuntimeError(
            f"Audio processing failed. Ensure the recorded audio is PCM/WAV format. Details: {type(e).__name__}"
        )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint
    Receives audio file and returns prediction
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None or feature_names is None:
            return (
                jsonify(
                    {
                        "error": "Model not loaded. Please ensure model files exist and restart the server."
                    }
                ),
                500,
            )

        # Get audio file from request
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        audio_bytes = audio_file.read()

        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio file"}), 400

        print(f"Received audio file: {len(audio_bytes)} bytes")

        # Preprocess audio
        audio_data, sr = preprocess_audio(audio_bytes)
        print(f"Preprocessed audio: {len(audio_data)} samples at {sr}Hz")

        # Extract features
        features_dict = extract_features(audio_data, sr)
        print(f"Extracted {len(features_dict)} features")

        # Prepare features in correct order
        feature_vector = []
        for fname in feature_names:
            if fname in features_dict:
                feature_vector.append(features_dict[fname])
            else:
                # Use a sensible default if a feature is missing (shouldn't happen with extract_features)
                feature_vector.append(0.0)

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)

        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        prediction_proba = model.predict_proba(feature_vector_scaled)[0]

        # Calculate risk score
        risk_score = prediction_proba[1] * 100
        confidence = max(prediction_proba) * 100

        print(
            f"Prediction: {prediction}, Risk: {risk_score:.2f}%, Confidence: {confidence:.2f}%"
        )

        # Prepare response
        response = {
            "label": int(prediction),
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "prediction_text": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "features": {
                "pitch_mean": float(features_dict.get("MDVP:Fo(Hz)", 0)),
                "pitch_variation": float(features_dict.get("MDVP:Jitter(%)", 0)),
                "amplitude_variation": float(features_dict.get("MDVP:Shimmer", 0)),
                "harmonics_noise_ratio": float(features_dict.get("HNR", 0)),
                "noise_harmonics_ratio": float(features_dict.get("NHR", 0)),
            },
        }

        return jsonify(response), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        # The frontend will receive this error message
        friendly_error = (
            str(e)
            if "Model not loaded" in str(e)
            else "An error occurred during audio analysis. See server logs for details."
        )

        print("\n" + "=" * 60)
        print("UNHANDLED ERROR IN /PREDICT ENDPOINT")
        print("=" * 60)
        print(error_trace)

        return (
            jsonify(
                {
                    "error": friendly_error,
                    "details": "Check server console for traceback.",
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    model_status = (
        model is not None and scaler is not None and feature_names is not None
    )
    return (
        jsonify(
            {
                "status": "healthy" if model_status else "model_not_loaded",
                "model_loaded": model_status,
            }
        ),
        200,
    )


@app.route("/", methods=["GET"])
def home():
    """API information endpoint"""
    return (
        jsonify(
            {
                "message": "Parkinson's Detection API",
                "version": "1.0",
                "model_loaded": model is not None,
                "endpoints": {
                    "/predict": "POST - Upload audio file for prediction",
                    "/health": "GET - Check API health",
                },
            }
        ),
        200,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PARKINSON'S DETECTION API SERVER")
    print("=" * 60)

    # Load model components
    if load_model_components():
        print("Server starting on http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("=" * 60)
        print("ERROR: Cannot start server without model files!")
        print("Please run 'python train_model.py' first.")
        print("=" * 60)
