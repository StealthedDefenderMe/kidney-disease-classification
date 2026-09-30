from src.cnn_classification.constants import CONFIG_FILE_PATH
from src.cnn_classification.utils.common import read_yaml
from src.cnn_classification.entity.config_entity import DataInjectionEntity
from pathlib import Path

class  ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)

    def get_data_injection_config(self) -> DataInjectionEntity:
        config = self.config.data_injection

        return DataInjectionEntity(
            source_url = config.source_url,
            root_dir = Path(config.root_dir)
        )