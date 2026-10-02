from src.cnn_classification.components.transformation import DataTransformation
from src.cnn_classification.config.configuration import ConfigurationManager

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()