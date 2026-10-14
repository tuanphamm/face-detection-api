FROM tuanphammds/face-detection-api:deploy_base.v1

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
