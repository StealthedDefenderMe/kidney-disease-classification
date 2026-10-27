from fastapi import APIRouter, Depends
from src.api.dto.prediction_dto import SinglePredictionDto
from src.api.modules.prediction.prediction_service import PredictionService

router = APIRouter( # Creates a group of related API routes, similar to a controller in NestJS.
    prefix="/api/v1/predictions", # Adds this prefix to every route defined in this controller.
    tags=["Prediction"], # Groups these endpoints under Prediction in Swagger UI (/docs), making the API documentation organized.
)


@router.post("/single")
async def predict_image(request: SinglePredictionDto = Depends()): 
    # Depends() is a dependency injection mechanism, similar to how NestJS injects services or other providers.
    # tells FastAPI to create SinglePredictionDto and resolve its required parameters.
    # Think of Depends() as a way to tell FastAPI, "Resolve this dependency for me before executing my endpoint.
    return await PredictionService.predict(request.file)
