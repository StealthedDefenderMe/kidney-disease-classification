from src.cnn_classification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnn_classification.utils.common import read_yaml
from src.cnn_classification.entity.config_entity import DataInjectionEntity, PrepareBaseModelEntity, DataTransformationEntity, TrainingEntity
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

    def get_data_transformation_config(self) -> DataTransformationEntity:
        config = self.config.data_transformation

        return DataTransformationEntity(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformed_image_size=self.params.TRANSFORMED_IMAGE_SIZE,
            batch_size=self.params.BATCH_SIZE,
            train_split=self.params.TRAIN_SPLIT,
            validation_split=self.params.VALIDATION_SPLIT,
            test_split=self.params.TEST_SPLIT,
            seed=self.params.SEED,
            augmentation=self.params.AUGMENTATION
        )

    def get_training_config(self) -> TrainingEntity:
        config = self.config.training

        return TrainingEntity(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            epochs=self.params.EPOCHS,
            image_size=self.params.IMAGE_SIZE
        )