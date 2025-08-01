import joblib
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

def examine_model():
    # Paths to model files
    joblib_path = r"C:\Users\Vishal\AiCryptodetect\AiCryptodetect\AiCryptodetect\random_forest_encryption_model.joblib"
    pkl_path = r"C:\Users\Vishal\AiCryptodetect\AiCryptodetect\AiCryptodetect\random_forest_encryption_model.pkl"
    
    print("=== ML Model Analysis ===\n")
    
    # Check file existence and sizes
    print("1. File Information:")
    for path, name in [(joblib_path, "Joblib"), (pkl_path, "Pickle")]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   {name} file: EXISTS ({size:,} bytes)")
        else:
            print(f"   {name} file: NOT FOUND")
    print()
    
    # Load and examine the model
    try:
        # Try loading with joblib first
        if os.path.exists(joblib_path):
            print("2. Loading model with joblib...")
            model = joblib.load(joblib_path)
        elif os.path.exists(pkl_path):
            print("2. Loading model with pickle...")
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print("ERROR: No model files found!")
            return
            
        print(f"   Model type: {type(model)}")
        print(f"   Model class: {model.__class__.__name__}")
        
        # Check if it's a RandomForest
        if isinstance(model, RandomForestClassifier):
            print(f"   Number of estimators: {model.n_estimators}")
            print(f"   Number of features: {model.n_features_in_}")
            print(f"   Feature names available: {hasattr(model, 'feature_names_in_')}")
            if hasattr(model, 'feature_names_in_'):
                print(f"   Feature names: {model.feature_names_in_[:10]}...")  # Show first 10
            print(f"   Number of classes: {len(model.classes_)}")
            print(f"   Classes: {model.classes_}")
        print()
        
        # Test the expected feature format
        print("3. Testing Feature Format:")
        expected_features = ['file_size', 'entropy'] + [f'byte_freq_{i}' for i in range(256)]
        print(f"   Expected features count: {len(expected_features)}")
        print(f"   Expected features: {expected_features[:5]}... + byte_freq_0 to byte_freq_255")
        
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ == len(expected_features):
                print("   ✓ Feature count matches expected format")
            else:
                print(f"   ✗ Feature count mismatch! Model expects {model.n_features_in_}, we provide {len(expected_features)}")
        print()
        
        # Test with sample data
        print("4. Testing with Sample Data:")
        try:
            # Create sample feature vector
            sample_features = {
                'file_size': 1000,
                'entropy': 7.5,
            }
            # Add byte frequencies (random for testing)
            for i in range(256):
                sample_features[f'byte_freq_{i}'] = np.random.randint(0, 10)
            
            # Convert to DataFrame
            df_test = pd.DataFrame([sample_features])
            df_test = df_test.reindex(columns=expected_features, fill_value=0)
            
            print(f"   Sample data shape: {df_test.shape}")
            print(f"   Sample data columns: {list(df_test.columns)[:5]}...")
            
            # Test prediction
            prediction = model.predict(df_test)
            print(f"   Sample prediction: {prediction[0]}")
            
            # Test prediction probabilities
            try:
                prediction_proba = model.predict_proba(df_test)
                max_prob = np.max(prediction_proba)
                print(f"   Sample prediction probability: {max_prob:.3f}")
                print(f"   All class probabilities: {prediction_proba[0]}")
            except Exception as e:
                print(f"   Prediction probability error: {e}")
                
        except Exception as e:
            print(f"   Sample prediction failed: {e}")
            print(f"   Error type: {type(e).__name__}")
        print()
        
        # Check model attributes
        print("5. Model Attributes:")
        important_attrs = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
        for attr in important_attrs:
            if hasattr(model, attr):
                print(f"   {attr}: {getattr(model, attr)}")
        print()
        
        return model
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_extraction():
    """Test the feature extraction functions from the Flask app"""
    print("6. Testing Feature Extraction Functions:")
    
    # Sample text data
    sample_text = b"This is a test file content for encryption analysis."
    
    try:
        # Calculate byte frequency
        byte_counts = np.bincount(np.frombuffer(sample_text, dtype=np.uint8), minlength=256)
        print(f"   Sample text length: {len(sample_text)} bytes")
        print(f"   Byte frequency array length: {len(byte_counts)}")
        print(f"   Non-zero byte frequencies: {np.count_nonzero(byte_counts)}")
        
        # Calculate entropy
        data_length = len(sample_text)
        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                probability = count / data_length
                entropy -= probability * np.log2(probability)
        
        print(f"   Calculated entropy: {entropy:.3f}")
        
        # Build feature vector
        features = {'file_size': len(sample_text), 'entropy': entropy}
        for i in range(256):
            features[f'byte_freq_{i}'] = byte_counts[i]
            
        print(f"   Feature vector length: {len(features)}")
        print(f"   Feature keys sample: {list(features.keys())[:5]}")
        
        return features
        
    except Exception as e:
        print(f"   Feature extraction error: {e}")
        return None

if __name__ == "__main__":
    model = examine_model()
    test_features = test_feature_extraction()
    
    if model and test_features:
        print("7. Final Integration Test:")
        try:
            expected_features = ['file_size', 'entropy'] + [f'byte_freq_{i}' for i in range(256)]
            df_test = pd.DataFrame([test_features])
            df_test = df_test.reindex(columns=expected_features, fill_value=0)
            
            prediction = model.predict(df_test)
            prediction_proba = model.predict_proba(df_test)
            
            print(f"   ✓ Integration test successful!")
            print(f"   Predicted class: {prediction[0]}")
            print(f"   Confidence: {np.max(prediction_proba):.3f}")
            
        except Exception as e:
            print(f"   ✗ Integration test failed: {e}")
