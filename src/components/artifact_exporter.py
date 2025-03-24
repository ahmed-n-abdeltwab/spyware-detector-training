import os
import shutil
import json
import pickle
from datetime import datetime
from typing import Dict, Any
from src.utils.directory import ensure_directory


class ArtifactExporter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Ensure base output directory exists with proper permissions
        self.config["output_dir"] = ensure_directory(
            os.path.abspath(config.get("output_dir", "release"))
        )

    def export_training_artifacts(
        self, training_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Export all artifacts from training pipeline with robust error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            release_dir = os.path.join(
                self.config["output_dir"], f"release_{timestamp}"
            )
            release_dir = ensure_directory(release_dir)

            exported_files = {}
            artifacts = training_results["artifacts"]

            # 1. Export model files with existence verification
            model_files_to_export = [
                ("model.pkl", "model"),
                ("metadata.json", "metadata"),
                ("metrics.json", "metrics"),
            ]

            model_dir = artifacts.get("model")
            if model_dir and os.path.exists(os.path.dirname(model_dir)):
                for filename, key in model_files_to_export:
                    src = os.path.join(os.path.dirname(model_dir), filename)
                    dest = os.path.join(release_dir, filename)
                    if self._safe_copy(src, dest):
                        exported_files[key] = dest
                    else:
                        # Create empty metrics.json if missing
                        if filename == "metrics.json":
                            self._create_default_metrics(release_dir, training_results)
                            exported_files["metrics"] = os.path.join(
                                release_dir, "metrics.json"
                            )

            # 2. Export preprocessing artifacts
            preprocess_artifacts = {
                "scaler.pkl": "scaler",
                "feature_selector.pkl": "feature_selector",
                "selected_features.json": "selected_features",
            }

            for artifact_file, artifact_key in preprocess_artifacts.items():
                src = artifacts.get(artifact_key)
                if src and os.path.exists(src):
                    dest = os.path.join(release_dir, artifact_file)
                    if self._safe_copy(src, dest):
                        exported_files[artifact_key] = dest

            # 3. Create feature structure file
            feature_structure = {
                "feature_names": training_results.get("selected_features", []),
                "required_features": len(training_results.get("selected_features", [])),
                "version": timestamp,
            }
            feature_structure_path = os.path.join(release_dir, "feature_structure.json")
            self._safe_write_json(feature_structure_path, feature_structure)
            exported_files["feature_structure"] = feature_structure_path

            # 4. Create package info
            self._create_package_info(release_dir, exported_files)

            return {k: os.path.abspath(v) for k, v in exported_files.items()}

        except Exception as e:
            print(f"Critical error during artifact export: {str(e)}")
            raise

    def _create_default_metrics(
        self, release_dir: str, training_results: Dict[str, Any]
    ):
        """Create default metrics file if missing"""
        default_metrics = training_results.get(
            "metrics",
            {
                "warning": "Metrics were not properly saved during training",
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [],
            },
        )
        metrics_path = os.path.join(release_dir, "metrics.json")
        self._safe_write_json(metrics_path, default_metrics)

    def _create_package_info(self, release_dir: str, files: Dict[str, str]):
        """Create documentation about package contents"""
        package_info = {
            "package_version": datetime.now().isoformat(),
            "contents": {
                "model": "Serialized trained model (pickle format)",
                "metadata": "Model training metadata and parameters",
                "metrics": "Model performance metrics",
                "feature_structure": "Required feature names and structure",
                "scaler": "Feature scaling parameters",
                "feature_selector": "Feature selection parameters",
            },
            "actual_contents": {k: os.path.basename(v) for k, v in files.items()},
            "notes": "Some components may be placeholders if export failed",
        }
        self._safe_write_json(
            os.path.join(release_dir, "package_info.json"), package_info
        )

    def _safe_copy(self, src: str, dest: str) -> bool:
        """Safe file copy with comprehensive error handling"""
        try:
            if os.path.exists(src):
                shutil.copy(src, dest)
                return os.path.exists(dest)
            return False
        except (OSError, shutil.SameFileError) as e:
            print(f"Warning: Failed to copy {src} to {dest}: {str(e)}")
            return False

    def _safe_write_json(self, path: str, data: Dict[str, Any]):
        """Atomic JSON file writing with comprehensive error handling"""
        temp_path = None
        try:
            temp_path = f"{path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                os.replace(temp_path, path)
            else:
                raise OSError("Temporary file is invalid")
        except (TypeError, ValueError) as e:
            print(f"JSON error writing to {path}: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        except OSError as e:
            print(f"Filesystem error writing to {path}: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        except Exception as e:
            print(f"Unexpected error writing to {path}: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
