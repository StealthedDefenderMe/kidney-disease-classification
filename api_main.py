from fastapi import FastAPI

from src.api.modules.prediction.prediction_controller import router

# creates your API application.
app = FastAPI(title="Kidney Disease Prediction API")

# registers your prediction endpoint with the application.
app.include_router(router)