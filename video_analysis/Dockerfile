FROM python:3.12-slim
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libgl1 ffmpeg libmagic1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model.onnx \
    MAX_UPLOAD_MB=512 \
    FRAME_SAMPLE_STRIDE=15 \
    FRAME_MAX_SAMPLES=64 \
    ENABLE_CLIP=1 \
    PREPROCESS_MODE=keras \
    CLIP_MODEL_NAME=ViT-B-32 \
    CLIP_PRETRAIN=laion2b_s34b_b79k
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt
COPY server.py infer.py analysis.py model.onnx /app/
EXPOSE 8888
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
