import joblib
import numpy as np
import json
from typing import Dict, Any
import os


if __name__ == "__main__":
    print('='*40)
    print("----- Sanity Check: Testing model and metadata loading")
    print('-'*40)
    model = joblib.load('model.pkl')
    metadata = json.load(open('model_metadata.json', 'r'))
    print("----- Model type:", type(model))
    print("----- Metadata keys:", metadata.keys())
    print('----- Features:', metadata.get('features', []))
    print('----- Making a test prediction with dummy data...')
    dummy_data = {feature: 0 for feature in metadata['features']}
    feature_values = [dummy_data[feature] for feature in metadata['features']]
    input_array = np.array(feature_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    print("----- Test prediction output:", prediction)
    print("----- It seems that the model and metadata are loaded correctly. but the prediction may not be meaningful with dummy data.")
    print("----- Sanity Check: Completed")
    print('='*40)
    print('='*40)


# Global variable to store the loaded model

model = None
metadata = None

def load_model_and_metadata():
    global model, metadata
    try:
        model = joblib.load('model.pkl')
        metadata = json.load(open('model_metadata.json', 'r'))
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False
    

    

def make_prediction(house_features: Dict[str, Any]) -> float: #This function take a dictionary as input
    global model , metadata

    if model is None:
        raise ValueError("Model not loaded")
    if metadata is None:
        raise ValueError("Model metadata not loaded")
    # Extract feature values in the correct order
    feature_values = [house_features[feature] for feature in metadata['features']]
    input_array = np.array(feature_values).reshape(1, -1) 
    prediction = model.predict(input_array)[0]
    
    return round(float(prediction), 2)

def get_model_info() -> Dict[str, Any]:
    global model , metadata
    if model is None:
        raise ValueError("Model not loaded")
    if metadata is None:
        raise ValueError("Model metadata not loaded")
    return metadata

def check_health() -> Dict[str, Any]:
    global model, metadata
    model_loaded = model is not None
    metadata_loaded = metadata is not None
    if model_loaded and metadata_loaded:
        health_status = {
            "status": "healthy",
            "model_loaded": True,
            "metadata_loaded": True,
            "message": "Model and metadata are loaded."
        }
    else:
        health_status = {
            "status": "unhealthy",
            "model_loaded": model_loaded,
            "metadata_loaded": metadata_loaded,
            "message": "Model or metadata not loaded."
        }
    return health_status
