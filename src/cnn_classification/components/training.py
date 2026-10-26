import tensorflow as tf
from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import TrainingEntity
from src.cnn_classification.utils.common import create_directories
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class Training:

    def __init__(self, config: TrainingEntity):
        self.config = config

    # We're receiving the datasets from transformation.
    def initiate_training(self, train_dataset, validation_dataset, test_dataset):

        logger.info("Started model training...")

        # Loading the already prepared base model
        model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # Compiling the model
        # This configures how the model learns:
        # Optimizer (adam) — adjusts model weights to improve predictions.
        # Loss — measures how wrong the predictions are. Your integer labels make sparse_categorical_crossentropy appropriate.
        # Accuracy — measures the percentage of correct predictions.
        # configures how the model will learn: optimizer, loss function, and metrics.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Here going to add model checkpoint
        # It automatically saves the best fit model whenever it achieves a new best validation loss during training
                # Create training directory before saving checkpoints
        create_directories([self.config.root_dir])

        # Save the model whenever validation loss improves
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.root_dir / "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        )


        # Training the model. This is where the actual learning happens.
        # train_dataset — model learns from images and updates its trainable weights.
        # validation_dataset — checks p erformance on images not used for weight updates.
        # epochs — how many times the model goes through the entire training dataset.
        # For example, EPOCHS: 1 means one pass through all training batches.
        # actually trains the model on training data and checks performance on validation data after each epoch
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.epochs,
            callbacks=[checkpoint, early_stopping]
        )

        # Load the best model selected using validation loss
        best_model = tf.keras.models.load_model(
            str(self.config.root_dir / "best_model.h5")
        )

        # Evaluate on unseen test data
        # evaluates the trained model on your test data.
        test_loss, test_accuracy = best_model.evaluate(test_dataset)

        # Collect actual labels and model predictions
        y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

        y_pred = np.argmax(best_model.predict(test_dataset), axis=1)

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        logger.info(f"Confusion Matrix:\n{cm}")

        # Generate per-class performance report
        class_names = ["Cyst", "Normal", "Stone", "Tumor"]

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=class_names,
            zero_division=0
        )

        logger.info(f"Classification Report:\n{report}")

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        # Create training directory if it doesn't exist
        # create_directories([self.config.root_dir])

        # Saving the trained model
        # Saves the model after training so you can load it later without training again.
        # model.save(self.config.trained_model_path)
        # best_model.save(self.config.trained_model_path)

        logger.info("Model training completed successfully.")