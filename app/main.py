from fastapi import FastAPI, Request
import uuid
from app.api.routes import router
from app.core.logging import setup_logging

setup_logging()

# api layer
# create app instance and include routes
app = FastAPI(title="Vegetable Classifier API")
app.include_router(router)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    return response


# @app.get("/health")
# def health():
#     '''
#     health check endpoint
#     '''
#     return {
#         "status": "ok"}