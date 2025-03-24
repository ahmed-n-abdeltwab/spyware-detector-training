import os
import pickle
import json
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from src.interfaces import IFeatureSelector
from src.utils.directory import ensure_directory


class FeatureSelector(IFeatureSelector):
    def __init__(self, config: dict):
        self.config = config
        self.selector = None
        self.selected_feature_indices = None
        self.selected_feature_names = None
        self.config["output_dir"] = ensure_directory(
            config.get("output_dir", "data/processed")
        )

    def select_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Select features based on configured method"""
        method = self.config.get("method", "mutual_info")
        k = min(self.config.get("k", 50), X.shape[1])

        if method == "mutual_info":
            self.selector = SelectKBest(mutual_info_classif, k=k)
        elif method == "chi2":
            # Ensure data is non-negative for chi2
            X = X - np.min(X, axis=0) if np.min(X) < 0 else X
            self.selector = SelectKBest(chi2, k=k)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        X_selected = self.selector.fit_transform(X, y)
        self.selected_feature_indices = self.selector.get_support(indices=True)

        # If feature names were provided in X (as DataFrame columns), use them
        if hasattr(X, "columns"):
            self.selected_feature_names = [
                X.columns[i] for i in self.selected_feature_indices
            ]
        else:
            self.selected_feature_names = [
                f"feature_{i}" for i in self.selected_feature_indices
            ]

        return X_selected, self.selected_feature_names

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection to new data"""
        if self.selector is None:
            raise RuntimeError("Selector not fitted. Call select_features() first.")
        return self.selector.transform(X)

    def save_artifacts(self) -> Dict[str, str]:
        """Save selector and feature names"""
        artifacts = {}
        output_dir = self.config.get("output_dir", "data/processed")

        if self.selector:
            selector_path = os.path.join(output_dir, "feature_selector.pkl")
            with open(selector_path, "wb") as f:
                pickle.dump(self.selector, f)
            artifacts["feature_selector"] = selector_path

        if self.selected_feature_names:
            features_path = os.path.join(output_dir, "selected_features.json")
            with open(features_path, "w") as f:
                json.dump(self.selected_feature_names, f)
            artifacts["selected_features"] = features_path

        return artifacts
