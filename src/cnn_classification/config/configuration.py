from src.cnn_classification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnn_classification.utils.common import read_yaml
from src.cnn_classification.entity.config_entity import DataInjectionEntity, PrepareBaseModelEntity
from pathlib import Path

class  ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_data_injection_config(self) -> DataInjectionEntity:
        config = self.config.data_injection

        return DataInjectionEntity(
            source_url = config.source_url,
            root_dir = Path(config.root_dir)
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelEntity:
        config = self.config.prepare_base_model

        return PrepareBaseModelEntity(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            image_size=self.params.IMAGE_SIZE,
            include_top=self.params.INCLUDE_TOP,
            classes=self.params.CLASSES,
            weights=self.params.WEIGHTS,
            dense_units=self.params.DENSE_UNITS
        )