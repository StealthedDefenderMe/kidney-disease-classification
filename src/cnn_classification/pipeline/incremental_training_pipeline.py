from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.incremental_training import IncrementalTraining
from src.cnn_classification.pipeline.incremental_data_transformation_pipeline import (
    IncrementalDataTransformationPipeline,
)


class IncrementalTrainingPipeline:

    def __init__(self):
        pass

    def initiate_incremental_training(self):

        # Step 1: Prepare training and validation datasets
        data_transformation_pipeline = IncrementalDataTransformationPipeline()

        train_dataset, validation_dataset = (
            data_transformation_pipeline.initiate_incremental_data_transformation()
        )

        # Step 2: Load incremental training configuration
        config = ConfigurationManager()
        training_config = config.get_incremental_training_config()

        # Step 3: Initialize incremental training
        incremental_training = IncrementalTraining(training_config)

        # Step 4: Continue training the existing model
        candidate_model_path = incremental_training.initiate_incremental_training(
            train_dataset,
            validation_dataset
        )

        return candidate_model_path