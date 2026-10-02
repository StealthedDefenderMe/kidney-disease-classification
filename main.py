from src.cnn_classification import logger
from src.cnn_classification.pipeline.injection_pipeline import DataInjectionPipeline
from src.cnn_classification.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline
from src.cnn_classification.pipeline.transformation_pipeline import DataTransformationPipeline

logger.info("Starting the application...")

try:
    # Data Injection Pipeline
    # pipeline = DataInjectionPipeline()
    # pipeline.main()

    # Preparing base model pipeline
    # base_model_pipeline = PrepareBaseModelPipeline()
    # base_model_pipeline.main()

    # Preparing data transformation pipeline
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.main()

    logger.info("Application finished running")

except Exception as e:
    logger.exception(e)
    raise