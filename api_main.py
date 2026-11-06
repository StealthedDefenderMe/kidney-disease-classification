from fastapi import FastAPI
from src.api.modules.train_input.train_input_controller import train_input_router
from src.api.modules.prediction.prediction_controller import router as prediction_router

# creates your API application.
app = FastAPI(title="Kidney Disease Prediction API")

# packing all controller routers together
routers = [
    prediction_router,
    train_input_router
]

# registering all controller routers with the application.
for router in routers:
    app.include_router(router)