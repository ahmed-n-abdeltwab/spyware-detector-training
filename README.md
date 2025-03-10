# Spyware Detector Training

## Overview
This repository contains the training pipeline for a spyware detection model. The pipeline includes **data processing, feature extraction, feature selection, model training, evaluation, and model exporting** for deployment.

## Features
- **Data Processing**: Loads and preprocesses malware data.
- **Feature Extraction**: Scales and transforms features for better model performance.
- **Feature Selection**: Selects the most relevant features using `mutual_info_classif`.
- **Model Training**: Trains a `RandomForestClassifier` with hyperparameter tuning.
- **Model Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix.
- **Model Exporting**: Saves trained models for deployment in a separate Flask API.

## Installation
### Prerequisites
- Python 3.8+
- Docker (for containerized execution)
- Virtual environment (optional but recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/ahmed-n-abdeltwab/spyware-detector-training.git
cd spyware-detector-training

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Run the Training Pipeline
```bash
python main.py
```
This will execute the following steps:
1. **Load & Process Data** (splitting into train, validation, and test sets)
2. **Feature Extraction & Selection**
3. **Train & Evaluate the Model**
4. **Save the Model, Scaler, and Feature Selector**

### Running with Docker
You can run the training pipeline inside a Docker container to ensure a consistent environment.

#### **Build the Docker Image**
```bash
docker build -t spyware-detector-training .
```

#### **Run the Training Pipeline in a Container**
```bash
docker run --rm -v $(pwd)/data:/app/data spyware-detector-training
```

This will mount the `data` directory so that processed files and models are saved locally.

## Model Deployment
The trained model is **not used directly in this repo**. Instead, it is used in a **Flask-based API** for predictions. The Flask API automatically fetches the latest trained model from GitHub.

To integrate with the API, push your trained model files to the GitHub repo:
```
models/
├── random_forest.pkl
├── scaler.pkl
├── feature_selector.pkl
```

## Updating the Model
1. **Train a new model** using `main.py`.
2. **Upload the new model files** to the `spyware-detector-models` GitHub repository.
3. The Flask API will **automatically update the model** when restarted.

## Contributing
Contributions are welcome! If you find a bug or want to improve the model, feel free to submit a pull request.

## License
This project is licensed under the MIT License.


