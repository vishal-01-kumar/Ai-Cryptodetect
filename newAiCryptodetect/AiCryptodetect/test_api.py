import requests
import json

# Test the Flask server connectivity
try:
    # Test basic connectivity
    response = requests.get('http://localhost:5000/')
    print(f"Server status: {response.status_code}")
    
    # Test text prediction endpoint
    test_data = {"text": "Hello, this is a test string for encryption analysis."}
    response = requests.post('http://localhost:5000/predict_text', 
                           json=test_data,
                           headers={'Content-Type': 'application/json'})
    
    print(f"Text prediction response status: {response.status_code}")
    print(f"Text prediction response: {response.json()}")
    
    # Test file prediction with a simple text file
    test_file_content = b"This is test file content for encryption analysis."
    files = {'file': ('test.txt', test_file_content, 'text/plain')}
    
    response = requests.post('http://localhost:5000/predict', files=files)
    print(f"File prediction response status: {response.status_code}")
    print(f"File prediction response: {response.json()}")
    
except Exception as e:
    print(f"Error testing API: {e}")
