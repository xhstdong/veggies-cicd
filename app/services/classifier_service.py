import time
import logging
import hashlib
from fastapi import HTTPException
from PIL import UnidentifiedImageError
from app.metrics import metrics

logger = logging.getLogger("classifier_service")

def classify_image(classifier, image_bytes, request_id=None):
    '''
    function that takes a classifier instance and image_bytes
    and returns a dictionary with class, confidence and latency.
    orchestration layer
    '''
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    metrics["total_requests"] += 1
    start = time.time()
    try:
        result = classifier.predict(image_bytes)
        latency = time.time() - start
        metrics["successful_predictions"] += 1
        metrics["total_latency"] += latency

        logger.info(
            f"request_id={request_id} "
            f"image_hash={image_hash} "
            f"prediction={result['class']} "
            f"confidence={result['confidence']:.4f} "
            f"latency={latency:.4f}s"
        )

        return {
            "prediction": result["class"],
            "confidence": result["confidence"],
            "latency": latency
        }
    except UnidentifiedImageError as e:
        logger.error(
            f"request_id={request_id} "
            f"image_hash={image_hash} "
            f"invalid_image error={str(e)}"
        )
        metrics["invalid_images"] += 1
        metrics["total_latency"] += latency
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
    except Exception as e:
        logger.error(
            f"request_id={request_id} "
            f"image_hash={image_hash} "
            f"error={str(e)}",
            exc_info=True
        )
        metrics["failures"] += 1
        metrics["total_latency"] += latency
        raise