from fastapi import APIRouter, File, Depends
from src.api.modules.train_input.train_input_service import TrainInputServiceClass
from src.api.dto.prediction_dto import SinglePredictionDto

train_input_service = TrainInputServiceClass()

# creates a separate router for training data collection
train_input_router = APIRouter(
    prefix="/api/v1/train-input",
    tags=["Train_input"]
)

@train_input_router.post("/upload")
async def upload_training_image(request: SinglePredictionDto = Depends()):
    return await train_input_service.save_image(request.file)