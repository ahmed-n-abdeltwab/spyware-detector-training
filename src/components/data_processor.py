import os
import pandas as pd
from typing import Tuple
from src.interfaces import IDataProcessor


class DataProcessor(IDataProcessor):
    def __init__(self, config: dict):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        required = ["data_path", "label_column"]
        if not all(k in self.config for k in required):
            raise ValueError(f"Missing required config keys: {required}")

    def load_data(self) -> pd.DataFrame:
        """Load data from configured source"""
        if not os.path.exists(self.config["data_path"]):
            raise FileNotFoundError(
                f"Data file not found at {self.config['data_path']}"
            )

        try:
            return pd.read_csv(
                self.config["data_path"],
                sep=self.config.get("separator", "\t"),
                engine="python",
            )
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")

    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and prepare data"""
        # Normalize column names
        data.columns = data.columns.astype(str).str.strip().str.lower()

        # Identify label column
        label_col = next(
            (col for col in data.columns if self.config["label_column"] in col.lower()),
            None,
        )
        if label_col is None:
            raise ValueError(f"Label column '{self.config['label_column']}' not found")

        # Ensure labels are numeric
        y = pd.to_numeric(data[label_col], errors="coerce")
        if y.isna().any():
            raise ValueError("Label column contains non-numeric values")

        X = data.drop(columns=[label_col])

        # Convert all features to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop completely null columns/rows
        X = X.dropna(axis=1, how="all").dropna(axis=0, how="all")

        return X, y
