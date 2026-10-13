from src.cnn_classification.components.training import Training
from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.pipeline.transformation_pipeline import DataTransformationPipeline

class TrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()

        # Getting the transformed data set from transformation_pipeline.py
        train_dataset, validation_dataset = DataTransformationPipeline().main()

        # Getting the training configuration
        training_config = config_manager.get_training_config()

        # Create training component
        data_training = Training(config=training_config)

        # Start model training
        data_training.initiate_training(train_dataset=train_dataset, validation_dataset=validation_dataset)