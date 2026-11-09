from pathlib import Path
from uuid import uuid4
import shutil
from fastapi import UploadFile, HTTPException
from src.cnn_classification.config.configuration import ConfigurationManager
from src.cnn_classification.pipeline.prediction_pipeline import PredictionPipeline

class TrainInputServiceClass:

    def __init__(self):
        self.config = ConfigurationManager().get_train_input_config()

    async def save_image(self, file: UploadFile):

        # Validates the uploaded filename.
        if not file.filename: raise HTTPException(status_code=400, detail="No file provided")

        # Validates the image extension.
        suffix = Path(file.filename).suffix.lower()

        if suffix not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, and PNG images are supported"
            )

        try:

            # Reads the uploaded image.
            content = await file.read()
            
            if not content: raise HTTPException(status_code=400, detail="Uploaded file is empty")

            PENDING_DIR = Path(self.config.pending_dir)

            # Creates the pending directory if it does not exist.
            PENDING_DIR.mkdir(parents=True, exist_ok=True)

            # Generates a unique filename to avoid overwriting images.
            image_name = f"{uuid4().hex}{suffix}"
            image_path = PENDING_DIR / image_name

            # Saves the uploaded image locally.
            image_path.write_bytes(content)

            pipeline = PredictionPipeline()
            prediction = pipeline.predict(str(image_path))

            # Gets prediction class and confidence
            predicted_class = prediction["predicted_class"]
            confidence = prediction["confidence"]

            # checks if the prediction meets our configured threashold
            if confidence >= self.config.confidence_threshold :

                # Creates a separate directory for predicted class
                class_dir = Path(self.config.pseudo_labeled_dir) / predicted_class
                class_dir.mkdir(parents=True, exist_ok=True)

                # moves image from pending to pseudo-labeled folder
                destination = class_dir / image_path.name
                shutil.move(str(image_path), str(destination))

                status = "pseudo_labeled"

            else:
                status = "pending"

            return {
                "message": "Image collected successfully",
                "image_id": image_path.stem,
                "status": status,
                "prediction": prediction
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Failed to save training image"
            ) from e

        finally:
            await file.close()