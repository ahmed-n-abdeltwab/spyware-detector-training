import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Handles loading, cleaning, and preprocessing of the malware dataset.
    """

    def __init__(self, data_dir="data", dataset_name="malwares.csv"):
        self.feature_columns = None
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, dataset_name)
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_data(self):
        """
        Load malware dataset from file and clean column names.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        # Try reading with tab separator first
        try:
            print("Attempting to read with tab separator...")
            data = pd.read_csv(self.dataset_path, sep="\t", engine="python")
        except pd.errors.ParserError:
            # If tab fails, try comma separator
            try:
                print(
                    "Tab separator failed. Attempting to read with comma separator..."
                )
                data = pd.read_csv(self.dataset_path, sep=",", engine="python")
            except pd.errors.ParserError:
                # If both fail, try reading without specifying separator
                print(
                    "Comma separator failed. Attempting to read with default settings..."
                )
                data = pd.read_csv(self.dataset_path, engine="python")

        # Check if the data was read correctly
        if data.shape[1] == 1:
            # If only one column is detected, try re-parsing with different settings
            print(
                "Warning: Data appears incorrectly formatted. Attempting to reprocess."
            )
            data = pd.read_csv(
                self.dataset_path,
                sep="\t",
                engine="python",
                header=0,
                skipinitialspace=True,
                quoting=3,
            )

        # Normalize column names
        data.columns = data.columns.astype(str).str.strip().str.lower()

        # Check for variations of 'labels' column
        label_col = next((col for col in data.columns if "label" in col.lower()), None)
        if label_col is None:
            raise ValueError(
                f"Missing 'labels' column in dataset. Available columns: {list(data.columns)}"
            )

        data.rename(columns={label_col: "labels"}, inplace=True)

        print(f"Loaded dataset with shape: {data.shape}")
        print("Dataset Columns:", list(data.columns))
        print("First few rows:\n", data.head())
        return data

    def preprocess_data(self, data):
        """
        Preprocess the dataset by extracting features and labels.
        """
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Invalid dataset provided. Expected a DataFrame.")

        if "labels" not in data.columns:
            raise ValueError(
                f"Error: 'labels' column not found. Available columns: {list(data.columns)}"
            )

        X = data.drop(columns=["labels"])
        y = data["labels"].astype(int)  # Ensure labels are integers

        # Debugging: Print first few rows of X
        print("First few rows of X before conversion:\n", X.head())

        # Ensure numeric conversion (convert non-numeric columns to NaN)
        X = X.apply(pd.to_numeric, errors="coerce")

        # Drop columns where all values are NaN (non-numeric columns)
        X = X.dropna(axis=1, how="all")

        # Drop rows where all features are NaN (fully missing rows)
        X = X.dropna(axis=0, how="all")

        print("Processed dataset: Features shape", X.shape, "Labels shape", y.shape)
        return X, y

    def split_dataset(self, X, y, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Split dataset into training, validation, and test sets.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise ValueError("Expected X as DataFrame and y as Series")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
            stratify=y_train_val,
        )

        print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Save processed datasets to disk.
        """
        if any(
            v is None or not isinstance(v, (pd.DataFrame, pd.Series))
            for v in [X_train, X_val, X_test, y_train, y_val, y_test]
        ):
            raise ValueError("Cannot save processed data. Some splits are invalid.")

        X_train.to_csv(os.path.join(self.processed_dir, "X_train.csv"), index=False)
        X_val.to_csv(os.path.join(self.processed_dir, "X_val.csv"), index=False)
        X_test.to_csv(os.path.join(self.processed_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.processed_dir, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(self.processed_dir, "y_val.csv"), index=False)
        y_test.to_csv(os.path.join(self.processed_dir, "y_test.csv"), index=False)

        print("Processed dataset saved successfully.")

    def save_feature_columns(self):
        import json

        path = os.path.join(self.processed_dir, "feature_columns.json")
        with open(path, "w") as f:
            json.dump(self.feature_columns, f)
