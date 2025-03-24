from typing import Dict, Any
import yaml
from src.registry import ComponentRegistry
from src.components.artifact_exporter import ArtifactExporter


class TrainingPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.components = ComponentRegistry().load_from_config(
            self.config["components"]
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def run(self) -> Dict[str, Any]:
        artifacts = {}

        # 1. Data Processing
        raw_data = self.components["data_processor"].load_data()
        X, y = self.components["data_processor"].preprocess(raw_data)

        # 2. Feature Extraction
        self.components["feature_extractor"].fit(X)
        X_features = self.components["feature_extractor"].transform(X)
        artifacts.update(self.components["feature_extractor"].save_artifacts())

        # 3. Feature Selection
        X_selected, selected_features = self.components[
            "feature_selector"
        ].select_features(X_features, y)
        artifacts.update(self.components["feature_selector"].save_artifacts())

        # 4. Model Training
        model = self.components["model_trainer"].train(X_selected, y)
        metrics = self.components["model_trainer"].evaluate(X_selected, y)
        artifacts.update(self.components["model_trainer"].save_model())

        # 5. Export artifacts
        exporter_config = {"output_dir": self.config.get("release_dir", "release")}
        exporter = ArtifactExporter(exporter_config)
        exported_files = exporter.export_training_artifacts(
            {
                "artifacts": artifacts,
                "selected_features": selected_features,
                "metrics": metrics,
            }
        )

        return {
            "model": model,
            "metrics": metrics,
            "artifacts": artifacts,
            "exported_files": exported_files,
            "selected_features": selected_features,
        }
