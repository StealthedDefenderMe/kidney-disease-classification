from src.cnn_classification import logger
from src.cnn_classification.components.injection import DataInjection
from src.cnn_classification.config.configuration import ConfigurationManager

logger.info("Starting the application...")

config_manager = ConfigurationManager() # responsible for loading the configuration.
data_injection_config = config_manager.get_data_injection_config() # Our config.yaml contains configuration for potentially many stages
# & We only want the configuration related to Data Injection.

obj = DataInjection(config=data_injection_config)
obj.initiate_data_injection()