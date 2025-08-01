import joblib
import numpy as np
import pandas as pd

# Load the model
joblib_path = r"C:\Users\Vishal\AiCryptodetect\AiCryptodetect\AiCryptodetect\random_forest_encryption_model.joblib"
model = joblib.load(joblib_path)

print("=== Model Feature Analysis ===\n")

print("Model expects these features in this exact order:")
print(f"Total features: {len(model.feature_names_in_)}")
print("Feature names:")
for i, feature_name in enumerate(model.feature_names_in_):
    print(f"  {i:3d}: {feature_name}")

print("\n" + "="*50)
print("Current Flask app provides:")
current_features = ['file_size', 'entropy'] + [f'byte_freq_{i}' for i in range(256)]
for i, feature_name in enumerate(current_features):
    print(f"  {i:3d}: {feature_name}")

print("\n" + "="*50)
print("Comparison:")
model_features = list(model.feature_names_in_)
matches = True
for i, (model_feat, current_feat) in enumerate(zip(model_features, current_features)):
    match = "✓" if model_feat == current_feat else "✗"
    if model_feat != current_feat:
        matches = False
    print(f"  {i:3d}: {match} Model: '{model_feat}' vs Current: '{current_feat}'")

if matches:
    print("\n✓ All features match!")
else:
    print("\n✗ Feature mismatch found!")
    
print(f"\nCorrect feature order for Flask app:")
print("FEATURE_COLUMNS = [")
for i, feature in enumerate(model.feature_names_in_):
    if i < len(model.feature_names_in_) - 1:
        print(f"    '{feature}',")
    else:
        print(f"    '{feature}'")
print("]")
