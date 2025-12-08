from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import magic
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def load_session(model_path: Optional[str] = None) -> ort.InferenceSession:
    path = model_path or os.environ.get("MODEL_PATH", "/app/model.onnx")

    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def sniff_mime(data: bytes, filename: Optional[str], content_type: Optional[str]) -> str:
    if content_type:
        if content_type.startswith("image/"): return "image"
        if content_type.startswith("video/"): return "video"

    try:
        kind = magic.from_buffer(data[:8192], mime=True) or ""
        if kind.startswith("image/"): return "image"
        if kind.startswith("video/"): return "video"

    except Exception:
        pass

    if filename:
        lower = filename.lower()

        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")):
            return "image"

        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
            return "video"
    return "image"

def _exif_transpose(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def _parse_input_layout_and_size(sess: ort.InferenceSession) -> Tuple[str, int, int]:
    inp = sess.get_inputs()[0]
    shape = inp.shape
    target_h = target_w = 128
    layout = "NHWC"
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        dims = []
        for d in shape:
            try:
                dims.append(int(d))

            except Exception:
                dims.append(None)
        if dims[-1] == 3:
            layout = "NHWC"
            h, w = dims[1], dims[2]
        elif dims[1] == 3:
            layout = "NCHW"
            h, w = dims[2], dims[3]
        else:
            h, w = None, None
        if isinstance(h, int) and h > 0: target_h = h
        if isinstance(w, int) and w > 0: target_w = w

    if os.environ.get("IMG_SIZE"):
        try:
            parts = os.environ["IMG_SIZE"].lower().replace("x", " ").split()
            if len(parts) == 1:
                target_h = target_w = int(parts[0])
            elif len(parts) >= 2:
                target_w = int(parts[0]); target_h = int(parts[1])

        except Exception:
            pass
    return layout, int(target_h), int(target_w)

def preprocess_pil_for_session(img: Image.Image, sess: ort.InferenceSession, mode: Optional[str] = None) -> np.ndarray:
    layout, tgt_h, tgt_w = _parse_input_layout_and_size(sess)
    mode = (mode or os.environ.get("PREPROCESS_MODE", "keras")).lower()
    img = _exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = max(tgt_w / max(1, w), tgt_h / max(1, h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h))
    left = max(0, (new_w - tgt_w) // 2)
    top = max(0, (new_h - tgt_h) // 2)
    img = img.crop((left, top, left + tgt_w, top + tgt_h))
    x = np.asarray(img).astype(np.float32)

    if mode == "keras":
        x = x / 255.0
    elif mode == "imagenet":
        x = (x / 255.0 - IM_MEAN) / IM_STD
    else:
        x = x / 255.0
    if layout == "NHWC":
        x = x[None, ...]
    else:
        x = np.transpose(x, (2, 0, 1))[None, ...]

    return x

def infer_image_bytes(sess: ort.InferenceSession, data: bytes) -> float:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()

    except Exception as e:
        try:
            img = Image.open(io.BytesIO(bytes(data)))
            img.load()
        
        except Exception:
            raise ValueError(f"Unsupported image data: {e}")

    x = preprocess_pil_for_session(img, sess)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], {in_name: x})[0]
    out = np.asarray(out)

    if out.ndim == 2 and out.shape[1] == 1:
        prob = float(out[0, 0])
    elif out.ndim == 2 and out.shape[1] == 2:
        prob = float(_softmax(out)[0, 1])
    else:
        prob = float(np.squeeze(out))

    return float(max(0.0, min(1.0, prob)))
