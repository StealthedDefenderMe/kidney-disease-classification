from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.components.prediction import Prediction

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.prediction_config = config.get_prediction_config()

    def predict(self, image_path):
        predictor = Prediction(
            model_path=self.prediction_config.model_path,
            class_names=self.prediction_config.class_names,
            image_size=self.prediction_config.image_size
        )

        return predictor.predict(image_path)