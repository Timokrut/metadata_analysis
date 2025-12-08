from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
try:
    import torch
    import open_clip
except Exception:
    torch = None
    open_clip = None

def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
def detect_spikes(frames: List[Image.Image], fps: float) -> Dict[str, Any]:
    events, mags = [], []
    last_event_ts = -1.0
    if not frames or len(frames) < 2:
        return {"score": 0.0, "anomaly": False, "tags": [], "events": []}
    gray_frames = [_to_gray(f) for f in frames]
    for idx in range(len(gray_frames) - 1):
        prev, nxt = gray_frames[idx], gray_frames[idx + 1]
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2).mean()
        mags.append(mag)
    if len(mags) < 5:
        return {"score": 0.0, "anomaly": False, "tags": [], "events": []}
    mu, sigma = float(np.mean(mags)), float(np.std(mags) + 1e-6)
    for idx, mag in enumerate(mags):
        ts = round((idx + 0.5) / fps, 2)
        if ts < last_event_ts + 1.0:
            continue
        z = (mag - mu) / sigma
        if z > 2:
            event_meta = {"z": round(z, 2)}
            try:
                s = ssim(gray_frames[idx], gray_frames[idx + 1])
                if s < 0.97:
                    event_meta["ssim"] = round(s, 3)
            except Exception:
                pass
            events.append({
                "module": "flow",
                "event": "flow_spike",
                "ts": ts,
                "dur": 0.0,
                "meta": event_meta
            })
            last_event_ts = ts
    return {
        "score": 0.10 if events else 0.0,
        "anomaly": bool(events),
        "tags": ["flow_spike"] if events else [],
        "events": events
    }

def detect_lighting_jumps_local(frames: List[Image.Image], fps: float) -> Dict[str, Any]:
    if not frames or len(frames) < 3:
        return {"score": 0.0, "anomaly": False, "tags": [], "events": []}
    v_vals = []
    for f in frames:
        arr = np.array(f)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        v_vals.append(hsv[..., 2].mean())
    v_vals = np.array(v_vals)
    diffs = np.abs(np.diff(v_vals))
    mu, sigma = float(np.mean(diffs)), float(np.std(diffs) + 1e-6)
    events = []
    last_event_ts = -1.0
    for i, d in enumerate(diffs):
        ts = round((i + 1) / fps, 2)
        if ts < last_event_ts + 1.0:
            continue
        z = (d - mu) / sigma
        if z > 3.0 and d > max(10.0, mu + 2 * sigma):
            events.append({
                "module": "lighting",
                "event": "lighting_jump",
                "ts": ts,
                "dur": 0.0,
                "meta": {"delta": round(float(d), 2), "z": round(float(z), 2)}
            })
            last_event_ts = ts
    return {
        "score": 0.10 if events else 0.0,
        "anomaly": bool(events),
        "tags": ["lighting_jump"] if events else [],
        "events": events
    }

REAL_PERSON_PROMPTS_CLIP = [
    "a typical frame from a live-action video recording of a real person",
    "a natural, unedited video still of a human being",
    "a person with natural skin texture and realistic lighting in a video",
]
FAKE_PERSON_PROMPTS_CLIP = [
    "an AI generated deepfake face with unnatural features in a video",
    "a digitally altered face, a manipulated facial video frame",
    "a video still of a person with synthetic-looking skin or odd facial morphing",
    "a face with mismatched lighting or blurry artifacts typical of video deepfakes",
    "eyes that look glassy, unfocused, or move unnaturally in a deepfake video",
    "an unnaturally smooth face, puppet-like movements, or a face that seems digitally overlaid",
]
_cached_clip_text_features: Dict[str, Dict[str, Any]] = {}
CLIP_SCORE_QUANTILE = 0.95
CLIP_SCORE_SCALE = 8.0
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))
def maybe_load_clip():
    if open_clip is None or torch is None:
        return None, None, None
    from os import environ
    if environ.get("ENABLE_CLIP", "1") != "1":
        return None, None, None
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    model_name = environ.get("CLIP_MODEL_NAME", "ViT-B-32")
    pretrained = environ.get("CLIP_PRETRAIN", "laion2b_s34b_b79k")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess, device
def calculate_visual_clip_score(pil_frames: List[Image.Image], clip_model, clip_preprocess_fn, device: str) -> float:
    if not pil_frames or clip_model is None:
        return 0.0
    global _cached_clip_text_features
    cache_key = f"{id(clip_model)}_{device}"
    if cache_key not in _cached_clip_text_features:
        with torch.no_grad():
            real_tokens = open_clip.tokenize(REAL_PERSON_PROMPTS_CLIP).to(device)
            fake_tokens = open_clip.tokenize(FAKE_PERSON_PROMPTS_CLIP).to(device)
            cache = {}
            cache['real'] = clip_model.encode_text(real_tokens)
            cache['fake'] = clip_model.encode_text(fake_tokens)
            cache['real'] /= cache['real'].norm(dim=-1, keepdim=True)
            cache['fake'] /= cache['fake'].norm(dim=-1, keepdim=True)
            _cached_clip_text_features[cache_key] = cache
    real_text_features = _cached_clip_text_features[cache_key]['real']
    fake_text_features = _cached_clip_text_features[cache_key]['fake']
    all_image_features = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(pil_frames), batch_size):
            batch = pil_frames[i:i+batch_size]
            images_tensor = torch.stack([clip_preprocess_fn(frame) for frame in batch]).to(device)
            img_features = clip_model.encode_image(images_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            all_image_features.append(img_features)
    if not all_image_features:
        return 0.0
    import torch as _t
    feats = _t.cat(all_image_features)
    real_sims = feats @ real_text_features.T
    fake_sims = feats @ fake_text_features.T
    avg_real_sim_per_frame = real_sims.mean(dim=1)
    max_fake_sim_per_frame = fake_sims.max(dim=1).values
    differential = max_fake_sim_per_frame - avg_real_sim_per_frame
    if differential.numel() == 0:
        return 0.0
    scaled = differential.quantile(CLIP_SCORE_QUANTILE).item() * CLIP_SCORE_SCALE
    return _sigmoid(float(scaled))

def fuse_scores(meso4: float, clip_score: Optional[float], flow_score: Optional[float], lighting_score: Optional[float]) -> float:
    weights = {
        "meso4": 0.6,
        "clip": 0.25,
        "flow": 0.10,
        "lighting": 0.05,
    }
    scores = {
        "meso4": meso4,
        "clip": 0.0 if clip_score is None else clip_score,
        "flow": 0.0 if flow_score is None else flow_score,
        "lighting": 0.0 if lighting_score is None else lighting_score,
    }
    active = {k: v for k, v in scores.items() if v is not None}
    norm = sum(weights[k] for k in active.keys())
    if norm <= 0:
        return float(meso4)
    agg = sum((weights[k] * scores[k]) for k in active.keys()) / norm
    return float(max(0.0, min(1.0, agg)))
