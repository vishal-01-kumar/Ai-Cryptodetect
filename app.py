from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import pickle
import os
import math
import numpy as np
from PIL import Image
import io
import base64
import wave
import struct

try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Only WAV files will be supported for audio steganography.")

app = Flask(__name__, template_folder='newAiCryptodetect/AiCryptodetect/templates')

# --- Model Loading ---
# Define the paths to the saved model files
MODEL_JOBLIB_PATH = os.path.join("newAiCryptodetect", "AiCryptodetect", "AiCryptodetect", "random_forest_encryption_model.joblib")
MODEL_PKL_PATH = os.path.join("newAiCryptodetect", "AiCryptodetect", "AiCryptodetect", "random_forest_encryption_model.pkl")

model = None
# Attempt to load the model, prioritizing joblib
try:
    if os.path.exists(MODEL_JOBLIB_PATH):
        model = joblib.load(MODEL_JOBLIB_PATH)
        print(f"Model loaded successfully from {MODEL_JOBLIB_PATH}")
    elif os.path.exists(MODEL_PKL_PATH):
        with open(MODEL_PKL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PKL_PATH}")
    else:
        print("Warning: No trained model file found. Using dummy predictions.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Warning: Model loading failed. Using dummy predictions.")

# Define the expected feature columns (should match training data)
FEATURE_COLUMNS = ['file_size'] + [f'byte_freq_{i}' for i in range(256)] + ['entropy']

# --- Feature Extraction Functions ---
def calculate_byte_frequency(data):
    """Calculates the frequency of each byte (0-255) in binary data."""
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    return byte_counts.tolist()

def calculate_entropy(data):
    """Calculates the Shannon entropy of binary data."""
    if not data:
        return 0.0

    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    data_length = len(data)
    entropy = 0.0

    for count in byte_counts:
        if count > 0:
            probability = count / data_length
            entropy -= probability * math.log2(probability)

    return entropy

def extract_features(file_content):
    """Extracts features from binary file content."""
    features = {}
    
    # 1. File size/length
    features['file_size'] = len(file_content)
    
    # 2. Byte frequency distribution
    byte_frequencies = calculate_byte_frequency(file_content)
    for i in range(256):
        features[f'byte_freq_{i}'] = byte_frequencies[i]
    
    # 3. Entropy
    features['entropy'] = calculate_entropy(file_content)
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file content
        file_content = file.read()
        
        # Extract features
        features = extract_features(file_content)
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])[FEATURE_COLUMNS]
        
        # Make prediction
        if model is not None:
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0]
            
            return jsonify({
                'prediction': 'Encrypted' if prediction == 1 else 'Not Encrypted',
                'confidence': float(max(probability)),
                'file_size': features['file_size'],
                'entropy': features['entropy']
            })
        else:
            # Dummy prediction when model is not available
            return jsonify({
                'prediction': 'Encrypted' if features['entropy'] > 7.0 else 'Not Encrypted',
                'confidence': 0.85,
                'file_size': features['file_size'],
                'entropy': features['entropy']
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel deployment
if __name__ == '__main__':
    app.run(debug=True)
