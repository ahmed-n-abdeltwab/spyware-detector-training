import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    Extracts relevant features from the malware dataset for spyware detection.
    """
    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train):
        """Fit the scaler on training data."""
        self.scaler.fit(X_train)
        self.is_fitted = True
    
    def transform(self, X):
        """Transform data using the fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def save_scaler(self):
        """Save the scaler."""
        import pickle
        os.makedirs(self.output_dir, exist_ok=True)
        scaler_path = f"{self.output_dir}/scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    def load_scaler(self):
        """Load the scaler."""
        import pickle
        scaler_path = f"{self.output_dir}/scaler.pkl"
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            print(f"Scaler loaded from {scaler_path}")
        except FileNotFoundError:
            print("Scaler file not found.")
