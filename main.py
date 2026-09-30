from src.cnn_classification import logger
from src.cnn_classification.pipeline.injection_pipeline import DataInjectionPipeline

logger.info("Starting the application...")

try:
    pipeline = DataInjectionPipeline()
    pipeline.main()
except Exception as e:
    logger.exception(e)
    raise