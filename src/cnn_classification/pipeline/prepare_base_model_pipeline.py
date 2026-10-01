from src.cnn_classification.components.prepare_base_model import PrepareBaseModel
from src.cnn_classification.config.configuration import ConfigurationManager

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager() # Responsible for loading the config
        base_model_config = config_manager.get_prepare_base_model_config()
        # Our config.yaml contains configuration for potentially many stages
        # & We only want the configuration related to prepare base model.

        obj = PrepareBaseModel(config=base_model_config)
        obj.prepare_full_model()