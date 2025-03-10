import os
import pickle
import json
import pandas as pd

class ModelLoader:
    """
    Handles dynamic loading of trained models.
    """
    def __init__(self, models_dir='models/saved', processed_dir='data/processed'):
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.current_model = None
        self.scaler = None
        self.feature_selector = None
    
    def list_models(self):
        """
        List all available model versions.
        """
        if not os.path.exists(self.models_dir):
            print("No models directory found.")
            return []
        return [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]
    
    def load_model(self, model_version=None):
        """
        Load a specific model version. If not specified, load the latest.
        """
        models = self.list_models()
        if not models:
            print("No saved models available.")
            return False
        
        # Use the latest model if none is specified
        if model_version is None:
            model_version = sorted(models, reverse=True)[0]
        
        # Load scaler and feature selector
        scaler_path = os.path.join(self.processed_dir, 'scaler.pkl')
        selector_path = os.path.join(self.processed_dir, 'feature_selector.pkl')
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)
            print("Loaded scaler and feature selector.")
        except FileNotFoundError as e:
            print(f"Error loading components: {e}")
            return False
        return True
    
    def predict(self, X_raw):
        """Predict on raw data."""
        if None in [self.current_model, self.scaler, self.feature_selector]:
            print("Load model, scaler, and selector first.")
            return None
        
        X_scaled = self.scaler.transform(X_raw)
        X_selected = self.feature_selector.transform(X_scaled)
        return self.current_model.predict(X_selected)
    
    def predict_proba(self, X):
        """
        Make probability predictions if the model supports it.
        """
        if self.current_model is None:
            print("No model loaded. Call load_model first.")
            return None
        
        if hasattr(self.current_model, 'predict_proba'):
            return self.current_model.predict_proba(X)
        else:
            print("The loaded model does not support probability predictions.")
            return None

