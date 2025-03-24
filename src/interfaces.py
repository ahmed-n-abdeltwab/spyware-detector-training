from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class IDataProcessor(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class IFeatureExtractor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save_artifacts(self) -> Dict[str, str]:
        pass


class IFeatureSelector(ABC):
    @abstractmethod
    def select_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        pass

    @abstractmethod
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save_artifacts(self) -> Dict[str, str]:
        pass


class IModelTrainer(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Any:
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def save_model(self) -> Dict[str, str]:
        pass
