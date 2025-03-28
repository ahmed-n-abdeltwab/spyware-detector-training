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

        # Ensure release directory exists with proper permissions
        release_dir = os.path.abspath("release/latest")
        os.makedirs(release_dir, exist_ok=True)
        os.chmod(release_dir, 0o777)
        logger.info(f"Release directory: {release_dir}")
        logger.info(
            f"Directory permissions: {oct(os.stat(release_dir).st_mode & 0o777)}"
        )

        # Initialize and run pipeline
        pipeline = TrainingPipeline("config/pipeline.yaml")
        results = pipeline.run()

        # Verify exported files exist
        logger.info("Verifying exported files:")
        missing_files = []
        for name, path in results["exported_files"].items():
            if path and os.path.exists(path):
                logger.info(f"  - {name}: {path}")
                os.chmod(path, 0o644)
            else:
                logger.error(f"  - {name}: FILE MISSING")
                missing_files.append(name)

        if missing_files:
            raise FileNotFoundError(
                f"Missing exported files: {', '.join(missing_files)}"
            )

        # Log results
        logger.info("✅ Training completed successfully")
        logger.info(f"📊 Model Metrics:\n{json.dumps(results['metrics'], indent=2)}")

        # Create compressed release package
        archive_path = os.path.join(release_dir, "model_release.zip")
        shutil.make_archive(archive_path[:-4], "zip", release_dir)
        logger.info(f"📦 Created release package: {archive_path}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
