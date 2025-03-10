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

# --- feature_selection.py ---
class FeatureSelector:
    """
    Selects the most relevant features for spyware detection
    """
    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.selector = None
        self.selected_feature_indices = None
    
    def select_features(self, X_train, y_train, method='mutual_info', k=None):
        """
        Select top k features based on the specified method
        """
        # If k is not specified, use half of the features or 50, whichever is smaller
        if k is None:
            k = min(50, X_train.shape[1] // 2)
            
        print(f"Selecting top {k} features using {method} method")
        
        # Choose the selection method
        if method == 'chi2':
            # Ensure non-negative values
            X_train_for_selection = X_train - np.min(X_train, axis=0)
            selector = SelectKBest(chi2, k=min(k, X_train.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
            X_train_for_selection = X_train
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
        
        # Fit the selector
        X_selected = selector.fit_transform(X_train_for_selection, y_train)
        
        # Save the selected feature indices
        self.selector = selector
        self.selected_feature_indices = np.where(selector.get_support())[0]
        
        # Print the names of the selected features
        feature_names = X_train.columns[self.selected_feature_indices].tolist()
        print(f"Selected {X_selected.shape[1]} features out of {X_train.shape[1]}")
        print(f"Top 10 selected features: {feature_names[:10]}")
        
        return X_selected, feature_names
    
    def transform_features(self, X):
        """
        Transform data using the previously fitted selector
        """
        if self.selector is None:
            print("Feature selector not fitted yet. Call select_features first.")
            return X
        
        X_selected = self.selector.transform(X)
        print(f"Transformed features shape: {X_selected.shape}")
        return X_selected
    
    def save_selector(self):
        """
        Save the feature selector for future use
        """
        selector_path = os.path.join(self.output_dir, 'feature_selector.pkl')
        with open(selector_path, 'wb') as f:
            pickle.dump(self.selector, f)
        
        indices_path = os.path.join(self.output_dir, 'selected_feature_indices.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(self.selected_feature_indices, f)
        
        print(f"Feature selector saved to {selector_path}")
    
    def load_selector(self):
        """
        Load the saved feature selector
        """
        selector_path = os.path.join(self.output_dir, 'feature_selector.pkl')
        indices_path = os.path.join(self.output_dir, 'selected_feature_indices.pkl')
        
        if os.path.exists(selector_path) and os.path.exists(indices_path):
            with open(selector_path, 'rb') as f:
                self.selector = pickle.load(f)
            
            with open(indices_path, 'rb') as f:
                self.selected_feature_indices = pickle.load(f)
                
            print(f"Feature selector loaded from {selector_path}")
            return True
        else:
            print(f"Feature selector files not found")
            return False
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores if available
        """
        if self.selector is None:
            print("Feature selector not fitted yet.")
            return None
        
        if hasattr(self.selector, 'scores_'):
            # Get indices of selected features
            selected_indices = np.where(self.selector.get_support())[0]
            
            # Get scores for the selected features
            scores = self.selector.scores_[selected_indices]
            
            # Get feature names if provided
            if feature_names is not None:
                if isinstance(feature_names, pd.Index):
                    feature_names = feature_names.values
                feature_names = np.array(feature_names)[selected_indices]
                feature_importance = {feature_names[i]: scores[i] for i in range(len(scores))}
            else:
                feature_importance = {f"feature_{selected_indices[i]}": scores[i] for i in range(len(scores))}
            
            # Sort by importance score
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        else:
            print("Feature importance scores not available.")
            return None

