from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from inference_engine import InferenceEngine

import tempfile
import os
import time
import cv2
import logging


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Create API application
app = FastAPI()


# Load model (the heart of everything - model is in here)
engine = InferenceEngine(
    "./runs/detect/runs/yolo11s_v2_img800/weights/best.pt"
)


# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# Metrics
TOTAL_REQUESTS = 0
TOTAL_PREDICT_REQUESTS = 0
TOTAL_IMAGE_REQUESTS = 0
FAILED_REQUESTS = 0
TOTAL_INFERENCE_TIME_MS = 0


@app.get("/")                           # GET request - is for fetching information/reading data/checking status
def root():
    return {"message": "API is running"}


@app.get("/health")                     #Health checkpoint (this step is for users to check if the API is still working)
def health():
    return {"message": "healthy"}


@app.get("/metrics")
def metrics():

    avg_inference = 0

    successful_requests = (
        TOTAL_PREDICT_REQUESTS +
        TOTAL_IMAGE_REQUESTS
    )

    if successful_requests > 0:
        avg_inference = round(
            TOTAL_INFERENCE_TIME_MS /
            successful_requests,
            2
        )

    return {
        "total_requests": TOTAL_REQUESTS,
        "predict_requests": TOTAL_PREDICT_REQUESTS,
        "image_requests": TOTAL_IMAGE_REQUESTS,
        "failed_requests": FAILED_REQUESTS,
        "avg_inference_ms": avg_inference
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):

    global TOTAL_REQUESTS
    global TOTAL_PREDICT_REQUESTS
    global FAILED_REQUESTS
    global TOTAL_INFERENCE_TIME_MS

    TOTAL_REQUESTS += 1
    TOTAL_PREDICT_REQUESTS += 1

    logger.info(
        f"Prediction request received: {file.filename}"
    )

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

        logger.error(
            f"Invalid content type: {file.content_type}"
        )

        FAILED_REQUESTS += 1

        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG files are allowed"
        )

    contents = file.file.read()

    if len(contents) > MAX_FILE_SIZE:

        FAILED_REQUESTS += 1

        raise HTTPException(
            status_code=413,
            detail="File size exceeds 10 MB limit"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(contents)
        temp_path = temp_file.name

    try:

        start_time = time.time()

        result = engine.predict(temp_path)

        end_time = time.time()

        result["inference_time_ms"] = round(
            (end_time - start_time) * 1000,
            2
        )

        result["inference_time_s"] = round(
            end_time - start_time,
            2
        )

        TOTAL_INFERENCE_TIME_MS += result[
            "inference_time_ms"
        ]

        logger.info(
            f"Inference completed in "
            f"{result['inference_time_ms']} ms"
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        FAILED_REQUESTS += 1

        logger.exception(
            f"Prediction failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):

    global TOTAL_REQUESTS
    global TOTAL_IMAGE_REQUESTS
    global FAILED_REQUESTS
    global TOTAL_INFERENCE_TIME_MS

    TOTAL_REQUESTS += 1
    TOTAL_IMAGE_REQUESTS += 1

    logger.info(
        f"Image prediction request received: "
        f"{file.filename}"
    )

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:

        logger.error(
            f"Invalid content type: {file.content_type}"
        )    

        FAILED_REQUESTS += 1

        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG files are allowed"
        )

    contents = file.file.read()

    if len(contents) > MAX_FILE_SIZE:

        FAILED_REQUESTS += 1

        raise HTTPException(
            status_code=413,
            detail="File size exceeds 10 MB limit"
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(contents)
        temp_path = temp_file.name

    try:

        start_time = time.time()

        results = engine.predict_raw(
            temp_path
        )

        annotated_img = results[0].plot()

        end_time = time.time()

        inference_time_ms = round(
            (end_time - start_time) * 1000,
            2
        )

        TOTAL_INFERENCE_TIME_MS += (
            inference_time_ms
        )

        logger.info(
            f"Image annotation completed in "
            f"{inference_time_ms} ms"
        )

        success, buffer = cv2.imencode(
            ".jpg",
            annotated_img
        )

        if not success:

            FAILED_REQUESTS += 1

            raise HTTPException(
                status_code=500,
                detail="Failed to encode prediction image"
            )

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg"
        )

    except HTTPException:
        raise

    except Exception as e:

        FAILED_REQUESTS += 1

        logger.exception(
            f"Image prediction failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
    
    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.middleware("http")                         #production feature - FastAPI Middleware
async def request_timer(request, call_next):
    start = time.time()

    response = await call_next(request)

    duration = round(
        (time.time() - start) * 1000,
        2
    )

    logger.info(
        f"{request.method} {request.url.path} "
        f"| {response.status_code} "
        f"| {duration} ms"
    )

    return response

from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    logger.exception(
        f"Unhandled exception on "
        f"{request.method} {request.url.path}: "
        f"{str(exc)}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "path": request.url.path
        }
    )

