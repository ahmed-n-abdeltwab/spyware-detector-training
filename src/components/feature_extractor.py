import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
from src.interfaces import IFeatureExtractor
from src.utils.directory import ensure_directory


class FeatureExtractor(IFeatureExtractor):
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler() if config.get("scale_features", True) else None
        self.feature_names: List[str] = []
        self.config["output_dir"] = ensure_directory(
            config.get("output_dir", "data/processed")
        )

    def fit(self, X: pd.DataFrame) -> None:
        """Fit the feature extractor on training data"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()

        if self.scaler:
            self.scaler.fit(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted extractor"""
        features = X.values if isinstance(X, pd.DataFrame) else X

        if self.scaler:
            if not hasattr(self.scaler, "scale_"):
                raise RuntimeError("Scaler not fitted. Call fit() first.")
            features = self.scaler.transform(features)

        return features

    def save_artifacts(self) -> Dict[str, str]:
        """Save scaler and feature names"""
        artifacts = {}
        output_dir = self.config.get("output_dir", "data/processed")

        if self.scaler:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            artifacts["scaler"] = scaler_path

        if self.feature_names:
            features_path = os.path.join(output_dir, "feature_names.json")
            with open(features_path, "w") as f:
                json.dump(self.feature_names, f)
            artifacts["feature_names"] = features_path

        return artifacts
