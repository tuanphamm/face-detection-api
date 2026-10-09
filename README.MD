# Face Detection API

FastAPI service for face detection using YOLO.

## Features
- Face detection
- REST API
- Docker support

## Run locally

pip install -r requirements.txt
python main.py

## Docker

docker build -t face-detection-api .
docker run -p 8000:8000 face-detection-api