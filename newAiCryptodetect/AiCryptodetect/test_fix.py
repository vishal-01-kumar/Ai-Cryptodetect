import joblib
import pandas as pd
import numpy as np
import math

# Load the model
model = joblib.load(r"C:\Users\Vishal\AiCryptodetect\AiCryptodetect\AiCryptodetect\random_forest_encryption_model.joblib")

# Use the corrected feature order
FEATURE_COLUMNS = ['file_size'] + [f'byte_freq_{i}' for i in range(256)] + ['entropy']

print("=== Testing Fixed Feature Order ===")
print(f"Model expects {len(model.feature_names_in_)} features")
print(f"We provide {len(FEATURE_COLUMNS)} features")
print(f"Feature order matches: {list(model.feature_names_in_) == FEATURE_COLUMNS}")

# Test with sample data
sample_text = b"This is a test file content for encryption analysis."

# Extract features (same as in Flask app)
def calculate_byte_frequency(data):
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    return byte_counts.tolist()

def calculate_entropy(data):
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
    features = {}
    features['file_size'] = len(file_content)
    
    byte_frequencies = calculate_byte_frequency(file_content)
    for i in range(256):
        features[f'byte_freq_{i}'] = byte_frequencies[i]
    
    features['entropy'] = calculate_entropy(file_content)
    return features

# Test the complete pipeline
try:
    features = extract_features(sample_text)
    df_for_prediction = pd.DataFrame([features])
    df_for_prediction = df_for_prediction.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    print(f"\nDataFrame shape: {df_for_prediction.shape}")
    print(f"DataFrame columns match model: {list(df_for_prediction.columns) == list(model.feature_names_in_)}")
    
    # Make prediction
    prediction = model.predict(df_for_prediction)
    prediction_proba = model.predict_proba(df_for_prediction)
    
    print(f"\n✓ SUCCESS! Prediction working!")
    print(f"Predicted algorithm: {prediction[0]}")
    print(f"Confidence: {np.max(prediction_proba):.3f}")
    print(f"All class probabilities: {prediction_proba[0]}")
    print(f"Available classes: {model.classes_}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
