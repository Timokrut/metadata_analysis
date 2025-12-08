from __future__ import annotations
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import cv2
import numpy as np
from infer import load_session, infer_image_bytes, sniff_mime, preprocess_pil_for_session
from analysis import (
    detect_spikes,
    detect_lighting_jumps_local,
    maybe_load_clip,
    calculate_visual_clip_score,
    fuse_scores,
)
app = FastAPI(title="Deepfake Detector API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = None
_clip_model = None
_clip_preprocess = None
_clip_device = None
def get_session():
    global _session
    if _session is None:
        _session = load_session()
    return _session
def get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None and os.environ.get("ENABLE_CLIP", "1") == "1":
        _clip_model, _clip_preprocess, _clip_device = maybe_load_clip()
    return _clip_model, _clip_preprocess, _clip_device
@app.get("/health")
async def health():
    return {"status": "ok"}
def _read_video_frames_from_bytes(data: bytes, every_n: int, max_frames: int):
    import tempfile
    frames: List[Image.Image] = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Unable to read video stream")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 25.0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every_n == 0:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    rgb = frame
                frames.append(Image.fromarray(rgb))
                if len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
    if not frames:
        raise ValueError("No frames decoded from video")
    return frames, float(fps)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), verbose: bool = Query(False)):
    try:
        content = await file.read()
        max_mb = int(os.environ.get("MAX_UPLOAD_MB", "512"))
        if len(content) > max_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Limit {max_mb}MB")
        kind = sniff_mime(content, file.filename, file.content_type)
        sess = get_session()
        meso4_prob: float
        flow_info = None
        light_info = None
        clip_score = None
        if kind == "video":
            stride = int(os.environ.get("FRAME_SAMPLE_STRIDE", "15"))
            max_frames = int(os.environ.get("FRAME_MAX_SAMPLES", "64"))
            frames, fps = _read_video_frames_from_bytes(content, every_n=stride, max_frames=max_frames)
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            probs = []
            for f in frames:
                x = preprocess_pil_for_session(f, sess)
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
                raise HTTPException(status_code=400, detail="No frames processed for inference")
            meso4_prob = float(sum(probs) / len(probs))
            flow_info = detect_spikes(frames, fps)
            light_info = detect_lighting_jumps_local(frames, fps)
            model, preprocess, device = get_clip()
            if model is not None:
                try:
                    clip_score = calculate_visual_clip_score(frames, model, preprocess, device)
                except Exception:
                    clip_score = None
        else:
            meso4_prob = float(infer_image_bytes(sess, content))
            model, preprocess, device = get_clip()
            if model is not None:
                try:
                    from io import BytesIO
                    img = Image.open(BytesIO(content)).convert("RGB")
                    clip_score = calculate_visual_clip_score([img], model, preprocess, device)
                except Exception:
                    clip_score = None
        flow_score = (float(flow_info.get("score", 0.0)) if flow_info else None)
        lighting_score = (float(light_info.get("score", 0.0)) if light_info else None)
        prob_ai = fuse_scores(meso4_prob, clip_score, flow_score, lighting_score)
        is_ai = bool(prob_ai > 0.5)
        if not verbose:
            return JSONResponse(content={
                "is_AI": is_ai,
                "probability_of_AI": round(float(prob_ai), 4)
            })
        else:
            details = {
                "meso4_prob": round(float(meso4_prob), 4),
                "clip_score": (round(float(clip_score), 4) if clip_score is not None else None),
                "flow": (flow_info or {"score": None, "anomaly": None, "tags": [], "events": []}),
                "lighting": (light_info or {"score": None, "anomaly": None, "tags": [], "events": []}),
                "input_kind": kind,
            }
            return JSONResponse(content={
                "is_AI": is_ai,
                "probability_of_AI": round(float(prob_ai), 4),
                "signals": details
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8888, reload=False)
