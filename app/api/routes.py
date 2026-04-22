from fastapi import APIRouter, UploadFile, File, Request
import torch

from app.model.loader import load_model
from app.model.inference import VegetableClassifier
from app.services.classifier_service import classify_image
from app.metrics import metrics

router = APIRouter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# instantiate model and classifier
# model loads once at startup instead of per request
model = load_model("weights/veggie_model_weights.pth", device)
classifier = VegetableClassifier(model, device)

# asynchronous endpoints
@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    '''
    endpoint to receive an image file and return the predicted class, confidence and latency.
    '''
    image_bytes = await file.read()
    return classify_image(classifier, image_bytes, request_id = request.state.request_id)

@router.get("/metrics")
def get_metrics():
    total = metrics["total_requests"]

    avg_latency = (
        metrics["total_latency"] / total
        if total > 0 else 0
    )

    return {
        **metrics,
        "avg_latency": round(avg_latency, 2)
    }

@router.get("/health")
def health():
     return {
        "status": "ok",
        "model_loaded": classifier is not None}