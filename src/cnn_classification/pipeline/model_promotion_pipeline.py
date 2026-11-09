from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.model_promotion import ModelPromotion
from src.cnn_classification.pipeline.candidate_evaluation_pipeline import (
    CandidateEvaluationPipeline,
)


class ModelPromotionPipeline:

    def __init__(self):
        pass

    def initiate_model_promotion(self):

        # First evaluate existing and candidate models
        evaluation_pipeline = CandidateEvaluationPipeline()
        evaluation_results = (
            evaluation_pipeline.initiate_candidate_evaluation()
        )

        # Load promotion configuration
        config_manager = ConfigurationManager()
        promotion_config = config_manager.get_model_promotion_config()

        # Promote only if evaluation criteria pass
        model_promotion = ModelPromotion(promotion_config)

        promoted = model_promotion.promote_candidate(evaluation_results)

        return promoted