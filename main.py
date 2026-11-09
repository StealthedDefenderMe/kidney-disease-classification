from src.cnn_classification import logger
# from src.cnn_classification.pipeline.injection_pipeline import DataInjectionPipeline
# from src.cnn_classification.pipeline.prepare_base_model_pipeline import PrepareBaseModelPipeline
# from src.cnn_classification.pipeline.transformation_pipeline import DataTransformationPipeline
# from src.cnn_classification.pipeline.training_pipeline import TrainingPipeline
# from src.cnn_classification.pipeline.evaluation_pipeline import EvaluationPipeline
# from src.cnn_classification.pipeline.prediction_pipeline import PredictionPipeline
# from src.cnn_classification.pipeline.incremental_training_pipeline import IncrementalTrainingPipeline
from src.cnn_classification.pipeline.candidate_evaluation_pipeline import CandidateEvaluationPipeline

logger.info("Starting the application...")

try:
    # Data Injection Pipeline
    # pipeline = DataInjectionPipeline()
    # pipeline.main()

    # Preparing base model pipeline
    # base_model_pipeline = PrepareBaseModelPipeline()
    # base_model_pipeline.main()

    # Preparing data transformation pipeline
    # data_transformation_pipeline = DataTransformationPipeline()
    # data_transformation_pipeline.main()

    # Training pipeline
    # training_pipeline = TrainingPipeline()
    # training_pipeline.main()

    # Evaluation Pipeline
    # evaluation_pipeline = EvaluationPipeline()
    # evaluation_pipeline.main()

    # image_path = "artifacts/data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (403).jpg"

    # prediction_pipeline = PredictionPipeline()
    # result = prediction_pipeline.predict(image_path)
    # print(result)

    # Running incremental training pipeline
    # incremental_training_pipeline = IncrementalTrainingPipeline()
    # candidate_model_path = incremental_training_pipeline.initiate_incremental_training()
    # print("Candidate model created:", candidate_model_path)

    # Running candidate evaluation pipeline
    evaluation_pipeline = CandidateEvaluationPipeline()
    result = evaluation_pipeline.initiate_candidate_evaluation()
    print(result)

    logger.info("Application finished running")

except Exception as e:
    logger.exception(e)
    raise