from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.candidate_evaluation import CandidateEvaluation
from src.cnn_classification.pipeline.transformation_pipeline import (
    DataTransformationPipeline,
)


class CandidateEvaluationPipeline:

    def __init__(self):
        pass

    def initiate_candidate_evaluation(self):

        # Load the original test dataset
        data_transformation_pipeline = DataTransformationPipeline()

        validation_dataset = (
            data_transformation_pipeline.initiate_validation_transformation()
        )

        # Get candidate evaluation configuration
        config_manager = ConfigurationManager()

        evaluation_config = (
            config_manager.get_candidate_evaluation_config()
        )

        # Evaluate and compare both models
        candidate_evaluation = CandidateEvaluation(evaluation_config)

        results = candidate_evaluation.compare_models(validation_dataset)

        return results
