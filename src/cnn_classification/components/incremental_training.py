import tensorflow as tf
from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import IncrementalTrainingEntity
from src.cnn_classification.utils.common import create_directories


class IncrementalTraining:

    def __init__(self, config: IncrementalTrainingEntity):
        self.config = config

    # Continues training an already trained model using old & new images
    def initiate_incremental_training(self, train_dataset, validation_dataset):

        logger.info("Incremental Model training started...")

        # Loading the existing training model instead of imageNet vgg16
        existing_model_path = self.config.existing_model_path

        model = tf.keras.models.load_model(existing_model_path)

        # Verify the existing model's trainable layers.
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                for sub_layer in layer.layers:
                    logger.info(
                        f"Layer: {sub_layer.name}, Trainable: {sub_layer.trainable}"
                    )

        # Inspect which layers will be updated during incremental training
        for layer in model.layers:
            logger.info(
                f"Layer: {layer.name}, Trainable: {layer.trainable}"
            )

        # Keeping the existing model's learned knowledge
        # Only the classification head & last few VGG16 layers will be trainable
        # This assumes the model was originally saved with those trainable layers
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )

        # Creating the directory where candidate model will be stored
        candidate_dir = self.config.candidate_model_dir
        create_directories([candidate_dir])

        # here saving the best candidate model based on the validation loss
        # this does not overwrite the existing best_model_v2.h5
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(candidate_dir / "candidate_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )

        # Stop training when validation loss stops improving
        # Restore the best weights from this training run.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )

        # Continue training the exisiting model on the combined dataset
        model.fit(
            train_dataset,
            validation_data = validation_dataset,
            epochs = self.config.epochs,
            callbacks=[checkpoint, early_stopping]
        )

        logger.info("Incremental model training completed successfully")

        return str(candidate_dir / "candidate_model.h5")