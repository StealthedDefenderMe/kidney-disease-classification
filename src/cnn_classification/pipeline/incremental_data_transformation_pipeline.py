from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.incremental_data_transformation import (
    IncrementalDataTransformation,
)


class IncrementalDataTransformationPipeline:

    def __init__(self):
        pass

    def initiate_incremental_data_transformation(self):

        # Load incremental transformation configuration
        config = ConfigurationManager()

        data_transformation_config = (
            config.get_incremental_data_transformation_config()
        )

        # Initialize incremental data transformation
        data_transformation = IncrementalDataTransformation(
            config=data_transformation_config
        )

        # Create training and validation datasets
        train_dataset, validation_dataset = (
            data_transformation.initiate_incremental_transformation()
        )
        # This returns new as images in training dataset (new + old)
        return train_dataset, validation_dataset