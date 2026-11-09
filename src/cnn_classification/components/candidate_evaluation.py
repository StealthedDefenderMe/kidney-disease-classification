
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import CandidateEvaluationEntity


class CandidateEvaluation:

    def __init__(self, config: CandidateEvaluationEntity):
        self.config = config

    # Evaluates a model on the original test dataset
    def evaluate_model(self, model_path, evaluation_dataset, model_name):

        logger.info(f"Evaluating {model_name}...")

        # Load the model
        model = tf.keras.models.load_model(model_path)

        # Calculate test loss and accuracy
        results = model.evaluate(
            evaluation_dataset,
            verbose=1,
            return_dict=True
        )

        # Get actual labels from the test dataset
        y_true = np.concatenate([
            labels.numpy()
            for _, labels in evaluation_dataset
        ])

        # Get model predictions
        predictions = model.predict(evaluation_dataset)

        # Convert probabilities into predicted class numbers
        y_pred = np.argmax(predictions, axis=1)

        # Generate classification report
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(self.config.class_names))),
            target_names=self.config.class_names,
            zero_division=0
        )

        # Generate confusion matrix
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(self.config.class_names)))
        )

        logger.info(f"{model_name} Test Loss: {results['loss']:.4f}")
        logger.info(f"{model_name} Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"{model_name} Classification Report:\n{report}")
        logger.info(f"{model_name} Confusion Matrix:\n{matrix}")

        return {
            "loss": results["loss"],
            "accuracy": results["accuracy"],
            "classification_report": report,
            "confusion_matrix": matrix
        }

    # Compare existing model against the newly trained candidate
    def compare_models(self, evaluation_dataset):

        # Evaluate the currently active model
        existing_results = self.evaluate_model(
            self.config.existing_model_path,
            evaluation_dataset,
            "Existing Model"
        )

        # Evaluate the newly trained candidate model
        candidate_results = self.evaluate_model(
            self.config.candidate_model_path,
            evaluation_dataset,
            "Candidate Model"
        )

        logger.info("Model comparison completed.")

        return {
            "existing_model": existing_results,
            "candidate_model": candidate_results
        }
