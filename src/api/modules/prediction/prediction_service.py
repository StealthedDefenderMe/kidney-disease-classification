import tempfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
from src.cnn_classification.pipeline.prediction_pipeline import PredictionPipeline

class PredictionService:

    @staticmethod
    async def predict(file: UploadFile):
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        suffix = Path(file.filename).suffix.lower()

        if suffix not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, and PNG images are supported",
            )

        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            pipeline = PredictionPipeline()
            result = pipeline.predict(temp_path)

            return result

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Image prediction failed",
            ) from e
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()