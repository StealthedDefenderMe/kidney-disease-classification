from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.data_splitter_od import DataSplitterOD

class DataSplitterODPipeline:

    def __init__(self):
        pass

    def main(self):

        # Load the object detection dataset splitting configuration.
        config_manager = ConfigurationManager()
        data_splitter_config = config_manager.get_data_splitter_config()

        # Initialize the dataset splitter with the loaded configuration.
        data_splitter = DataSplitterOD(config=data_splitter_config)

        # Create train, validation and test folders and copy the images.
        split_manifest = data_splitter.initiate_data_splitting()

        # Return the manifest containing the details of the generated splits.
        return split_manifest
