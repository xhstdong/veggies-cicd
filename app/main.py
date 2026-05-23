from fastapi import FastAPI, Request
import os
import uuid
import torch
from app.api.routes import router
from app.core.logging import setup_logging
from app.model.loader import load_model
from app.model.inference import VegetableClassifier

setup_logging()

# api layer
# create app instance and include routes
app = FastAPI(title="Vegetable Classifier API")
app.include_router(router)
app.state.classifier = None

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    return response


@app.on_event("startup")
async def startup_event():
    if os.getenv("TESTING") == 'true':
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(
        "weights/veggie_model_weights.pth",
        device
    )

    classifier = VegetableClassifier(model, device)

    app.state.classifier = classifier