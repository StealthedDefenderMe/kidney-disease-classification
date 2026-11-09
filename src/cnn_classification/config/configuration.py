from pathlib import Path
from src.cnn_classification.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.cnn_classification.utils.common import read_yaml
from src.cnn_classification.entity.config_entity import ( CandidateEvaluationEntity, DataInjectionEntity, IncrementalDataTransformationEntity, IncrementalTrainingEntity, ModelPromotionEntity, PrepareBaseModelEntity, 
                                                         DataTransformationEntity, TrainInputEntity, TrainingEntity, 
                                                         PredictionEntity, DataSplitterEntity )

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
            evaluation_dir=Path(config.evaluation_dir),
            misclassified_images_dir=Path(config.misclassified_images_dir),
            misclassified_predictions_path=Path(config.misclassified_predictions_path),
            all_predictions_path=Path(config.all_predictions_path),
            epochs=self.params.EPOCHS,
            image_size=self.params.IMAGE_SIZE,
            class_names=self.params.CLASS_NAMES,
        )

    def get_prediction_config(self) -> PredictionEntity:
        config = self.config.prediction

        return PredictionEntity(
            model_path=Path(config.model_path),
            class_names=self.params.CLASS_NAMES,
            image_size=self.params.PREDICTION_IMAGE_SIZE
        )

    def get_data_splitter_config(self) -> DataSplitterEntity:
        config = self.config.data_splitter

        return DataSplitterEntity(
            class_names=self.params.YOLO_CLASS_NAMES,
            root_dir=Path(config.root_dir),
            source_data_path=Path(config.source_data_path),
            images_dir=Path(config.images_dir),
            labels_dir=Path(config.labels_dir),
            manifest_path=Path(config.manifest_path),
            train_split=self.params.YOLO_TRAIN_SPLIT,
            validation_split=self.params.YOLO_VALIDATION_SPLIT,
            test_split=self.params.YOLO_TEST_SPLIT,
            seed=self.params.YOLO_SPLIT_SEED
        )

    def get_train_input_config(self) -> TrainInputEntity:
        config = self.config.train_input

        return TrainInputEntity(
            pending_dir=config.pending_dir,
            pseudo_labeled_dir=config.pseudo_labeled_dir,
            confidence_threshold=config.confidence_threshold
        )

    def get_incremental_training_config(self) -> IncrementalTrainingEntity:
        config = self.config.incremental_training

        return IncrementalTrainingEntity(
            existing_model_path=Path(config.existing_model_path),
            candidate_model_dir=Path(config.candidate_model_dir),
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            patience=config.patience,
            min_new_images=self.params.MIN_NEW_IMAGES
        )

    # Returns configuration for incremental data transformation
    def get_incremental_data_transformation_config(self) -> IncrementalDataTransformationEntity:

        config = self.config.incremental_data_transformation

        return IncrementalDataTransformationEntity(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            pseudo_labeled_dir=Path(config.pseudo_labeled_dir),
            batch_size=config.batch_size,
            transformed_image_size=tuple(config.transformed_image_size),
            augmentation=config.augmentation,
            class_names=self.params.CLASS_NAMES,
            min_new_images=self.params.MIN_NEW_IMAGES
        )

    def get_candidate_evaluation_config(self) -> CandidateEvaluationEntity:

        config = self.config.candidate_evaluation

        return CandidateEvaluationEntity(
            existing_model_path=Path(config.existing_model_path),
            candidate_model_path=Path(config.candidate_model_path),
            class_names=self.params.CLASS_NAMES
        )
    
    def get_model_promotion_config(self) -> ModelPromotionEntity:
        config = self.config.model_promotion

        return ModelPromotionEntity(
            active_model_path=Path(config.active_model_path),
            candidate_model_path=Path(config.candidate_model_path),
            backup_model_path=Path(config.backup_model_path),
        )