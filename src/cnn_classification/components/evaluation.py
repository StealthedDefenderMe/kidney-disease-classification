
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import TrainingEntity
from src.cnn_classification.components.misclassified_evaluation import (
    save_misclassified_images,
)


class Evaluation:

    def __init__(self, config: TrainingEntity):
        self.config = config

    def initiate_evaluation(self, test_dataset, test_image_paths):

        logger.info("Started model evaluation...")

        # Load the best model saved during training
        model = tf.keras.models.load_model(
            str(self.config.root_dir / "best_model_v2.h5")
        )

        # Evaluate model performance on unseen test data
        test_loss, test_accuracy = model.evaluate(test_dataset)

        # Get actual labels
        y_true = np.concatenate(
            [labels.numpy() for _, labels in test_dataset]
        )

        # Get predicted probabilities
        y_prob = model.predict(test_dataset, verbose=0)

        # Get predicted classes and confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidence = np.max(y_prob, axis=1) * 100

        # Save misclassified images and prediction CSV files
        save_misclassified_images(
            y_true=y_true,
            y_pred=y_pred,
            confidence=confidence,
            test_image_paths=test_image_paths,
            class_names=self.config.class_names,
            evaluation_dir=self.config.evaluation_dir,
            misclassified_images_dir=self.config.misclassified_images_dir,
            misclassified_predictions_path=self.config.misclassified_predictions_path,
            all_predictions_path=self.config.all_predictions_path
        )

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        logger.info(f"Confusion Matrix:\n{cm}")

        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=self.config.class_names,
            zero_division=0
        )

        logger.info(f"Classification Report:\n{report}")

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        logger.info("Model evaluation completed successfully.")
