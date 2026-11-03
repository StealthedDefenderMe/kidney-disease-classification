
from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.transformation import DataTransformation
from src.cnn_classification.components.evaluation import Evaluation


class EvaluationPipeline:

    def __init__(self):
        pass

    def main(self):

        config_manager = ConfigurationManager()

        # Get data transformation configuration
        data_transformation_config = config_manager.get_data_transformation_config()

        # Get the test dataset & image path
        data_transformation = DataTransformation(config=data_transformation_config)

        # Loading evaluation data
        test_dataset, test_image_path = data_transformation.initiate_evaluation_transformation()

        # Get training config
        training_config = config_manager.get_training_config()

        # Create evaluation component
        evaluation = Evaluation(config=training_config)

        evaluation.initiate_evaluation(
            test_dataset=test_dataset,
            test_image_paths=test_image_path
        )