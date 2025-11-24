from __future__ import annotations
import io
import os
import tempfile
from typing import Tuple
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2

IM_SIZE = 128
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)
  
def load_session(model_path: str | None = None) -> ort.InferenceSession:
    path = model_path or os.environ.get("MODEL_PATH", "/models/meso4.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess
  
def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IM_SIZE, IM_SIZE))
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    return x
  
def _run_p_fake(sess: ort.InferenceSession, x: np.ndarray) -> float:
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logits = sess.run([out_name], {in_name: x})[0]
    probs = _softmax(logits.astype(np.float32))
    return float(probs[0, 1])
  
def infer_image_bytes(sess: ort.InferenceSession, data: bytes) -> float:
    img = Image.open(io.BytesIO(data))
    x = preprocess_pil(img)
    return _run_p_fake(sess, x)
  
def infer_video_bytes(sess: ort.InferenceSession, data: bytes, every_n: int = 15, max_frames: int = 64) -> float:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Unable to read video stream")
        frames_seen = 0
        probs = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                x = preprocess_pil(img)
                p = _run_p_fake(sess, x)
                probs.append(p)
                frames_seen += 1
                if frames_seen >= max_frames:
                    break
            i += 1
        cap.release()
    if not probs:
        raise ValueError("No frames processed from video")
    return float(np.mean(probs))
  
def detect_mime(filename: str | None, content_type: str | None) -> str:
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
    if filename:
        lower = filename.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return "image"
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return "video"
    return "image"
