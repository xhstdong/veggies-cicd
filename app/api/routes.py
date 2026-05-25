from fastapi import APIRouter, UploadFile, File, Request

from app.services.classifier_service import classify_image
from app.metrics import metrics

router = APIRouter()

# asynchronous endpoints
@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    '''
    endpoint to receive an image file and return the predicted class, confidence and latency.
    '''
    image_bytes = await file.read()
    classifier = request.app.state.classifier
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
def health(request: Request):
     classifier = request.app.state.classifier
     return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "classifier_type": type(classifier).__name__
            if classifier else None}