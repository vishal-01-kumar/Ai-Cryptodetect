from flask import Flask, render_template, request, jsonify
import os
import math

app = Flask(__name__)

# Simplified version without heavy ML dependencies for Vercel deployment

# --- Feature Extraction Functions ---
def calculate_byte_frequency(data):
    """Calculates the frequency of each byte (0-255) in binary data."""
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    return byte_counts

def calculate_entropy(data):
    """Calculates the Shannon entropy of binary data."""
    if not data:
        return 0.0

    byte_counts = calculate_byte_frequency(data)
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
        
        # Simple entropy-based prediction (without ML model)
        # High entropy typically indicates encrypted/compressed data
        entropy = features['entropy']
        
        # Simple heuristic: entropy > 7.0 suggests encryption
        is_encrypted = entropy > 7.0
        confidence = min(0.95, (entropy / 8.0) if is_encrypted else (8.0 - entropy) / 8.0)
        
        return jsonify({
            'prediction': 'Encrypted' if is_encrypted else 'Not Encrypted',
            'confidence': round(confidence, 2),
            'file_size': features['file_size'],
            'entropy': round(entropy, 3)
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel deployment
if __name__ == '__main__':
    app.run(debug=True)
