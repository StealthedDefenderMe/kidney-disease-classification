from src.cnn_classification.constants import CONFIG_FILE_PATH
from src.cnn_classification.utils.common import read_yaml

class  ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)

    def get_data_injection_config(self):
        return self.config.data_injection