import pandas as pd
from src.data_processor import DataProcessor
from src.feature_extraction import FeatureExtractor
from src.feature_selection import FeatureSelector
from src.model_trainer import ModelTrainer
from src.model_loader import ModelLoader


def main():
    """
    Main function to run the spyware detector trainer pipeline with the malware dataset.
    """
    print("Starting Spyware Detector Trainer pipeline...")

    # Step 1: Data Processing
    print("\n=== Step 1: Data Processing ===")
    data_processor = DataProcessor(data_dir="data")

    # Load dataset
    data_path = "data/malwares.csv"
    data = pd.read_csv(data_path, sep="\t")

    # Preprocess dataset
    X, y = data_processor.preprocess_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_dataset(X, y)
    data_processor.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # Step 2: Feature Extraction
    print("\n=== Step 2: Feature Extraction ===")
    feature_extractor = FeatureExtractor()
    feature_extractor.fit(X_train)  # Fit only on training data
    X_train_features = feature_extractor.transform(X_train)
    X_val_features = feature_extractor.transform(X_val)
    X_test_features = feature_extractor.transform(X_test)
    feature_extractor.save_scaler()

    # Step 3: Feature Selection
    print("\n=== Step 3: Feature Selection ===")
    feature_selector = FeatureSelector()
    X_train_selected, feature_names = feature_selector.select_features(
        X_train_features, y_train, method='chi2', k=50
    )
    X_val_selected = feature_selector.transform_features(X_val_features)
    X_test_selected = feature_selector.transform_features(X_test_features)
    feature_selector.save_selector()

    # Step 4: Model Training
    print("\n=== Step 4: Model Training ===")
    model_trainer = ModelTrainer()
    trained_model = model_trainer.train_model(X_train_selected, y_train)
    model_trainer.evaluate_model(X_val_selected, y_val)
    model_trainer.save_model()

    # Step 5: Model Loading and Prediction
    print("\n=== Step 5: Model Loading and Prediction ===")
    model_loader = ModelLoader()
    if model_loader.load_model():
        # Example: Load new raw data (replace with actual new data)
        new_data = pd.read_csv("data/malwares.csv", sep="\t")

        # Preprocess new data (same as training)
        X_new, _ = data_processor.preprocess_data(new_data)

        # Predict
        predictions = model_loader.predict(X_new)
        print("Predictions:", predictions)


if __name__ == "__main__":
    main()
