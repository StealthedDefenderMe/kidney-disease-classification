from src.cnn_classification.components.transformation import DataTransformation
from src.cnn_classification.config.configuration import ConfigurationManager

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        train_dataset, validation_dataset = data_transformation.initiate_data_transformation()

        return train_dataset, validation_dataset

    def initiate_evaluation_transformation(self):

        config_manager = ConfigurationManager()

        data_transformation_config = config_manager.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)

        test_dataset, test_image_paths = (
            data_transformation.initiate_evaluation_transformation()
        )

        return test_dataset, test_image_paths

    def initiate_validation_transformation(self):

        config_manager = ConfigurationManager()
        config = config_manager.get_data_transformation_config()

        data_transformation = DataTransformation(config=config)

        split_datasets, _ = data_transformation.load_datasets(["validation"])

        return split_datasets["validation"]