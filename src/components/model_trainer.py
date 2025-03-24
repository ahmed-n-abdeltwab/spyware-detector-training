import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.interfaces import IModelTrainer
from src.utils.directory import ensure_directory


class ModelTrainer(IModelTrainer):
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.best_params = None
        self.metrics = None
        self.config["output_dir"] = ensure_directory(
            config.get("output_dir", "models/saved")
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model with hyperparameter tuning"""
        model_type = self.config.get("model_type", "random_forest")
        hyperparams = self.config.get("hyperparams", {})

        # Initialize model
        model = self._get_model_instance(model_type)

        # Hyperparameter tuning
        if hyperparams:
            start_time = time.time()
            grid_search = GridSearchCV(
                model, hyperparams, cv=3, scoring="f1_weighted", verbose=1
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            print(f"Hyperparameter tuning completed in {time.time() - start_time:.2f}s")
            print(f"Best parameters: {self.best_params}")
        else:
            self.model = model
            self.model.fit(X, y)

        return self.model

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X)

        self.metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        return self.metrics

    def save_model(self) -> Dict[str, str]:
        """Save trained model and metadata"""
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")

        artifacts = {}
        output_dir = self.config.get("output_dir", "models/saved")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(output_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        artifacts["model"] = model_path

        # Save metadata
        metadata = {
            "model_type": self.config.get("model_type"),
            "timestamp": timestamp,
            "hyperparameters": self.best_params,
            "metrics": self.metrics,
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = metadata_path

        return artifacts

    def _get_model_instance(self, model_type: str) -> Any:
        """Create model instance based on type"""
        if model_type == "random_forest":
            return RandomForestClassifier(random_state=42, class_weight="balanced")
        elif model_type == "svm":
            return SVC(probability=True, random_state=42, class_weight="balanced")
        elif model_type == "neural_net":
            return MLPClassifier(random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
