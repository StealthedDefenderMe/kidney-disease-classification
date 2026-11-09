import shutil
import os

from src.cnn_classification import logger
from src.cnn_classification.entity.config_entity import ModelPromotionEntity


class ModelPromotion:

    def __init__(self, config: ModelPromotionEntity):
        self.config = config

    def promote_candidate(self, evaluation_results):

        existing = evaluation_results["existing_model"]
        candidate = evaluation_results["candidate_model"]

        # Promotion criteria
        accuracy_improved = candidate["accuracy"] >= existing["accuracy"]
        loss_improved = candidate["loss"] <= existing["loss"]

        if not (accuracy_improved and loss_improved):
            logger.warning(
                "Candidate rejected. Existing model will remain active."
            )
            return False

        active_path = self.config.active_model_path
        candidate_path = self.config.candidate_model_path
        backup_path = self.config.backup_model_path

        # Make sure both model files exist
        if not active_path.exists():
            raise FileNotFoundError(f"Active model not found: {active_path}")

        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate model not found: {candidate_path}")

        # Prevent accidentally replacing the active model with itself
        if active_path.resolve() == candidate_path.resolve():
            raise ValueError("Active and candidate model paths must be different.")

        # Create backup directory
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Back up the current active model
        shutil.copy2(active_path, backup_path)

        # Copy candidate to a temporary file, then replace active model
        temporary_path = active_path.with_name(
            active_path.name + ".tmp"
        )

        try:
            shutil.copy2(candidate_path, temporary_path)
            os.replace(temporary_path, active_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        logger.info("Candidate model promoted successfully.")
        logger.info(f"Previous model backup saved at: {backup_path}")

        return True