import tensorflow as tf
from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import TrainingEntity

class Training:

    def __init__(self, config: TrainingEntity):
        self.config = config

    def initiate_training(self, train_dataset, validation_dataset):

        logger.info("Started model training...")

        # Loading the already prepared based model
        model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # Compiling the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )