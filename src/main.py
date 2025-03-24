import logging
import shutil
import sys
import os
import json
from src.pipeline import TrainingPipeline


def configure_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def main():
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 Starting Spyware Detector Training Pipeline")

        # Ensure release directory exists
        release_dir = os.path.abspath("release/latest")
        os.makedirs(release_dir, exist_ok=True)
        logger.info(f"Release directory: {release_dir}")

        # Initialize and run pipeline
        pipeline = TrainingPipeline("config/pipeline.yaml")
        results = pipeline.run()

        # Log results
        logger.info("✅ Training completed successfully")
        logger.info(f"📊 Model Metrics:\n{json.dumps(results['metrics'], indent=2)}")

        # Create compressed release package
        shutil.make_archive("model_release", "zip", release_dir)
        logger.info("📦 Created release package: model_release.zip")

        # Log exported files
        logger.info("💾 Artifacts generated:")
        for name, path in results["exported_files"].items():
            if path and os.path.exists(path):
                logger.info(f"  - {name}: {path}")
            else:
                logger.warning(f"  - {name}: FILE MISSING")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
