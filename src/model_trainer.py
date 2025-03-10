import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import pickle
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# --- model_trainer.py ---
class ModelTrainer:
    """
    Trains and evaluates machine learning models for spyware detection
    """
    def __init__(self, models_dir='models/saved'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.model = None
        self.model_type = None
        self.metrics = {}
        self.best_params = None
    
    def train_model(self, X_train, y_train, model_type='random_forest', hyperparams=None, cv=3):
        """
        Train a model of the specified type with hyperparameter tuning
        """
        self.model_type = model_type
        print(f"Training {model_type} model...")
        start_time = time.time()
        
        # Set default hyperparameters if none are provided
        if hyperparams is None:
            hyperparams = self._get_default_hyperparams(model_type)
        
        # Initialize the model based on the type
        base_model = self._get_model_instance(model_type)
        
        # Use GridSearchCV for hyperparameter tuning
        print(f"Performing hyperparameter tuning with {cv}-fold cross-validation")
        grid_search = GridSearchCV(base_model, hyperparams, cv=cv, scoring='f1_weighted', verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log results
        elapsed_time = time.time() - start_time
        print(f"Model training completed in {elapsed_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Store the best hyperparameters
        self.best_params = best_params
        
        return self.model
    
    def _get_default_hyperparams(self, model_type):
        """
        Get default hyperparameters for grid search
        """
        if model_type == 'random_forest':
            return {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif model_type == 'svm':
            return {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        elif model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,)],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['constant', 'adaptive']
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_model_instance(self, model_type):
        """
        Get a model instance based on the type
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=42, class_weight='balanced')
        elif model_type == 'svm':
            return SVC(probability=True, random_state=42, class_weight='balanced')
        elif model_type == 'neural_network':
            return MLPClassifier(random_state=42, max_iter=300)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def evaluate_model(self, X_val, y_val):
        """
        Evaluate the trained model on the validation set
        """
        if self.model is None:
            print("Model not trained yet. Call train_model first.")
            return None
        
        print("Evaluating model on validation set...")
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        # Generate detailed classification report
        report = classification_report(y_val, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Print summary
        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return self.metrics
    
    def feature_importance(self, feature_names):
        """
        Get feature importance from the trained model if available
        """
        if self.model is None:
            print("Model not trained yet. Call train_model first.")
            return None
        
        # Check if the model has feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Create a dictionary of feature names and their importance
            feature_importance = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    feature_importance[feature_names[i]] = importance
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            # Print top 10 features
            print("Top 10 important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                print(f"{i+1}. {feature}: {importance:.4f}")
            
            return feature_importance
        else:
            print("Feature importance not available for this model type.")
            return None
    
    def save_model(self, model_version=None):
        """
        Save the trained model and its metrics
        """
        if self.model is None:
            print("No model to save. Train a model first.")
            return
        
        # Generate a timestamp for the model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_version is None:
            model_version = f"{self.model_type}_{timestamp}"
        
        # Create a directory for this model version
        model_dir = os.path.join(self.models_dir, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save the best parameters
        if self.best_params:
            params_path = os.path.join(model_dir, 'best_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
        
        # Save the metrics
        if self.metrics:
            metrics_path = os.path.join(model_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
        
        # Create a metadata file
        metadata = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'version': model_version
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved successfully to {model_dir}")
        return model_dir
    
    def load_model(self, model_version=None):
        """
        Load a trained model from disk
        """
        # If no specific version is provided, find the latest one
        if model_version is None:
            versions = [d for d in os.listdir(self.models_dir) 
                      if os.path.isdir(os.path.join(self.models_dir, d))]
            if not versions:
                print("No saved models found.")
                return False
            
            # Sort versions by timestamp (assuming format model_type_YYYYMMDD_HHMMSS)
            versions.sort(reverse=True)
            model_version = versions[0]
        
        model_dir = os.path.join(self.models_dir, model_version)
        model_path = os.path.join(model_dir, 'model.pkl')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return False
        
        # Load the model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata if available
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.model_type = metadata.get('model_type', 'unknown')
            
            # Load metrics if available
            metrics_path = os.path.join(model_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            print(f"Model '{model_version}' loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

