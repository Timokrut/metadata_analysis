from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from analysis import (
    calculate_visual_clip_score,
    detect_lighting_jumps_local,
    detect_spikes,
    fuse_scores,
    maybe_load_clip,
)

from infer import (
    infer_image_bytes,
    load_session,
    preprocess_pil_for_session,
    sniff_mime,
)

# =========================================================
# CONFIG
# =========================================================

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "512"))
FRAME_SAMPLE_STRIDE = int(os.environ.get("FRAME_SAMPLE_STRIDE", "30"))
FRAME_MAX_SAMPLES = int(os.environ.get("FRAME_MAX_SAMPLES", "24"))
ENABLE_CLIP = os.environ.get("ENABLE_CLIP", "0") == "1"

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Deepfake Detector API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL MODELS
# =========================================================

_session = None
_clip_model = None
_clip_preprocess = None
_clip_device = None

# =========================================================
# MODEL LOADERS
# =========================================================


def get_session():
    global _session

    if _session is None:
        print("[INFO] Loading ONNX model...")
        _session = load_session()
        print("[INFO] ONNX model loaded")

    return _session



def get_clip():
    global _clip_model, _clip_preprocess, _clip_device

    if not ENABLE_CLIP:
        return None, None, None

    if _clip_model is None:
        print("[INFO] Loading CLIP model...")

        (
            _clip_model,
            _clip_preprocess,
            _clip_device,
        ) = maybe_load_clip()

        print("[INFO] CLIP model loaded")

    return _clip_model, _clip_preprocess, _clip_device


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }


# =========================================================
# VIDEO FRAME EXTRACTION
# =========================================================


def _read_video_frames(
    file_path: str,
    every_n: int,
    max_frames: int,
):
    """
    Extract sampled frames from video.
    """

    frames: List[Image.Image] = []

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or np.isnan(fps) or fps <= 0:
        fps = 25.0

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % every_n == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = frame

            frames.append(Image.fromarray(rgb))

            if len(frames) >= max_frames:
                break

        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError("No frames extracted")

    return frames, float(fps)


# =========================================================
# HEAVY ANALYSIS
# =========================================================


def run_video_analysis(
    file_path: str,
    verbose: bool = False,
):
    """
    Heavy CPU task.
    Runs in separate thread.
    """

    fpath = Path(file_path)

    if not fpath.exists():
        raise ValueError(f"File not found: {file_path}")

    file_size_mb = fpath.stat().st_size / (1024 * 1024)

    print(f"[INFO] Processing file: {fpath.name}")
    print(f"[INFO] File size: {file_size_mb:.2f} MB")

    if file_size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"File too large: {file_size_mb:.2f} MB")

    content = fpath.read_bytes()

    kind = sniff_mime(content, fpath.name, None)

    print(f"[INFO] Detected type: {kind}")

    sess = get_session()

    flow_info = None
    light_info = None
    clip_score = None

    # =====================================================
    # VIDEO
    # =====================================================

    if kind == "video":

        print("[INFO] Extracting frames...")

        frames, fps = _read_video_frames(
            str(fpath),
            every_n=FRAME_SAMPLE_STRIDE,
            max_frames=FRAME_MAX_SAMPLES,
        )

        print(f"[INFO] Extracted frames: {len(frames)}")
        print(f"[INFO] FPS: {fps}")

        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        probs = []

        for idx, frame in enumerate(frames):
            print(f"[INFO] Processing frame {idx + 1}/{len(frames)}")

            x = preprocess_pil_for_session(frame, sess)

            out = sess.run([out_name], {in_name: x})[0]

            out = np.asarray(out)

            if out.ndim == 2 and out.shape[1] == 1:
                p = float(out[0, 0])

            elif out.ndim == 2 and out.shape[1] == 2:
                from infer import _softmax
                p = float(_softmax(out)[0, 1])

            else:
                p = float(np.squeeze(out))

            p = max(0.0, min(1.0, p))

            probs.append(p)

        if not probs:
            raise ValueError("No predictions generated")

        meso4_prob = float(sum(probs) / len(probs))

        print(f"[INFO] Base model score: {meso4_prob:.4f}")

        # =================================================
        # ADDITIONAL SIGNALS
        # =================================================

        try:
            print("[INFO] Detecting motion anomalies...")
            flow_info = detect_spikes(frames, fps)
        except Exception as e:
            print(f"[WARNING] Flow detection failed: {e}")
            flow_info = None

        try:
            print("[INFO] Detecting lighting anomalies...")
            light_info = detect_lighting_jumps_local(frames, fps)
        except Exception as e:
            print(f"[WARNING] Lighting detection failed: {e}")
            light_info = None

        # =================================================
        # CLIP
        # =================================================

        if ENABLE_CLIP:
            try:
                model, preprocess, device = get_clip()

                if model is not None:
                    print("[INFO] Running CLIP analysis...")

                    clip_score = calculate_visual_clip_score(
                        frames,
                        model,
                        preprocess,
                        device,
                    )

            except Exception as e:
                print(f"[WARNING] CLIP failed: {e}")
                clip_score = None

    # =====================================================
    # IMAGE
    # =====================================================

    else:

        print("[INFO] Running image analysis...")

        meso4_prob = float(infer_image_bytes(sess, content))

        if ENABLE_CLIP:
            try:
                from io import BytesIO

                model, preprocess, device = get_clip()

                if model is not None:
                    img = Image.open(BytesIO(content)).convert("RGB")

                    clip_score = calculate_visual_clip_score(
                        [img],
                        model,
                        preprocess,
                        device,
                    )

            except Exception as e:
                print(f"[WARNING] CLIP failed: {e}")
                clip_score = None

    # =====================================================
    # FINAL SCORE
    # =====================================================

    flow_score = (
        float(flow_info.get("score", 0.0))
        if flow_info else None
    )

    lighting_score = (
        float(light_info.get("score", 0.0))
        if light_info else None
    )

    prob_ai = fuse_scores(
        meso4_prob,
        clip_score,
        flow_score,
        lighting_score,
    )

    is_ai = bool(prob_ai > 0.5)

    print(f"[INFO] Final probability: {prob_ai:.4f}")

    if not verbose:
        return {
            "is_AI": is_ai,
            "probability_of_ai": round(float(prob_ai), 4)
        }

    return {
        "is_AI": is_ai,
        "probability_of_ai": round(float(prob_ai), 4),
        "signals": {
            "meso4_prob": round(float(meso4_prob), 4),
            "clip_score": (
                round(float(clip_score), 4)
                if clip_score is not None else None
            ),
            "flow": flow_info,
            "lighting": light_info,
            "input_kind": kind,
        }
    }


# =========================================================
# API
# =========================================================

@app.post("/analyze")
async def analyze(
    file_path: str = Form(...),
    verbose: bool = Query(False),
):

    try:

        result = await asyncio.to_thread(
            run_video_analysis,
            file_path,
            verbose,
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        print(f"[ERROR] {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8888,
        reload=False,
        workers=1,
    )
