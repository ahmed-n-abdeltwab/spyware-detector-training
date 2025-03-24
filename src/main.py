import logging
import shutil
import sys
import os
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

        # Initialize and run pipeline
        pipeline = TrainingPipeline("config/pipeline.yaml")
        results = pipeline.run()

        # Log results
        logger.info("✅ Training completed successfully")
        logger.info(f"📊 Model Metrics:\n{json.dumps(results['metrics'], indent=2)}")

        # Get the actual release directory from exported files
        release_dir = next(
            (
                os.path.dirname(p)
                for p in results["exported_files"].values()
                if p and os.path.exists(os.path.dirname(p))
            ),
            None,
        )

        if not release_dir:
            raise RuntimeError("No valid release directory found in exported files")

        # Create compressed release package
        archive_base = os.path.basename(release_dir.rstrip("/"))
        shutil.make_archive(archive_base, "zip", release_dir)
        logger.info(f"📦 Created release package: {archive_base}.zip")

        logger.info("💾 Artifacts generated:")
        for name, path in results["exported_files"].items():
            logger.info(f"  - {name}: {path}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import json  # Import moved here to show it's needed

    main()
