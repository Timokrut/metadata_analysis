import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import io
import soundfile as sf
import librosa
import os
import logging

from importlib import import_module
from typing import Dict, Tuple, Any
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------------
# Логирование
# -----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Глобальные переменные
# -----------------------------

device = None
model = None

app = FastAPI(title="ASVspoof Detection System")

# -----------------------------
# API
# -----------------------------

@app.post("/analyze")
async def analyze_audio(file_path: str = Form(...)):
    """
    Анализ аудиофайла по пути в файловой системе
    """

    try:
        audio_data, sample_rate = read_audio_from_path(file_path)

        result = process_audio_data(
            audio_data,
            sample_rate
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception(f"Ошибка обработки файла {file_path}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -----------------------------
# Чтение аудио
# -----------------------------

def read_audio_from_path(file_path: str) -> Tuple[np.ndarray, int]:

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Файл не найден: {file_path}"
        )

    with open(file_path, "rb") as f:
        contents = f.read()

    try:
        audio_buffer = io.BytesIO(contents)

        audio_data, sample_rate = librosa.load(
            audio_buffer,
            sr=16000,
            mono=True
        )

        return audio_data, sample_rate

    except Exception as e:
        logger.warning(
            f"librosa не смог прочитать файл ({e}), пробуем soundfile"
        )

        audio_buffer = io.BytesIO(contents)

        audio_data, sample_rate = sf.read(audio_buffer)

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 16000:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000

        return audio_data, sample_rate

# -----------------------------
# Подготовка аудио
# -----------------------------

def pad_audio(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    x_len = len(x)

    if x_len >= max_len:
        return x[:max_len]

    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]

    return padded_x

# -----------------------------
# Акустические признаки
# -----------------------------

def extract_acoustic_features(
    audio_data: np.ndarray,
    sample_rate: int
) -> Dict:

    try:
        y = audio_data
        sr = sample_rate

        features = {}

        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        features["mfcc_mean"] = np.mean(
            mfccs,
            axis=1
        ).tolist()

        features["mfcc_std"] = np.std(
            mfccs,
            axis=1
        ).tolist()

        spec_cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )[0]

        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"] = float(np.std(spec_cent))

        spec_bw = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )[0]

        features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))

        rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )[0]

        features["spectral_rolloff_mean"] = float(np.mean(rolloff))

        flatness = librosa.feature.spectral_flatness(y=y)[0]

        features["spectral_flatness_mean"] = float(np.mean(flatness))

        rms = librosa.feature.rms(y=y)[0]

        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        zcr = librosa.feature.zero_crossing_rate(y)[0]

        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        features["amplitude_max"] = float(np.max(np.abs(y)))
        features["amplitude_mean"] = float(np.mean(np.abs(y)))
        features["amplitude_std"] = float(np.std(y))

        D = np.abs(librosa.stft(y))

        spectral_energy = np.sum(D, axis=0)

        spectral_entropy = -np.sum(
            spectral_energy *
            np.log(spectral_energy + 1e-12)
        )

        features["spectral_entropy"] = float(spectral_entropy)

        return features

    except Exception as e:
        logger.warning(
            f"Ошибка извлечения признаков: {e}"
        )

        return {}

# -----------------------------
# Обработка модели
# -----------------------------

def process_audio_data(
    audio_data: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:

    if model is None:
        raise RuntimeError("Модель не загружена")

    audio_padded = pad_audio(audio_data, 64600)

    input_tensor = torch.tensor(
        audio_padded,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():

        batch_coef, outputs = model(input_tensor)

        probs = F.softmax(outputs, dim=-1)

        score = float(probs[0, 1].item())

        embedding = batch_coef[0].cpu().numpy().tolist()

        coef = batch_coef[0].cpu().numpy()

        if len(coef) >= 160:

            T_max = coef[0:32].tolist()
            T_avg = coef[32:64].tolist()
            S_max = coef[64:96].tolist()
            S_avg = coef[96:128].tolist()
            master = coef[128:160].tolist()

        else:

            chunk_size = len(coef) // 5

            T_max = coef[0:chunk_size].tolist()
            T_avg = coef[chunk_size:2 * chunk_size].tolist()
            S_max = coef[2 * chunk_size:3 * chunk_size].tolist()
            S_avg = coef[3 * chunk_size:4 * chunk_size].tolist()

            if len(coef) >= chunk_size * 5:
                master = coef[4 * chunk_size:5 * chunk_size].tolist()
            else:
                master = []

        embedding_stats = {
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding)),
            "min": float(np.min(embedding)),
            "max": float(np.max(embedding))
        }

        acoustic_features = extract_acoustic_features(
            audio_data,
            sample_rate
        )

        confidence = float(
            abs(
                probs[0, 0].item() -
                probs[0, 1].item()
            )
        )

    return {
        "ai_probability": score,
        "predicted_class": "fake" if score > 0.5 else "real",
        "confidence": confidence,

        "T_max": T_max,
        "T_avg": T_avg,
        "S_max": S_max,
        "S_avg": S_avg,
        "master": master,

        "embedding_stats": embedding_stats,
        "embedding_size": len(embedding),

        "acoustic_features": acoustic_features,

        "embedding_preview": embedding[:20]
    }

# -----------------------------
# Загрузка модели
# -----------------------------

def get_model(
    model_config: Dict,
    device: torch.device
) -> torch.nn.Module:

    module = import_module(
        f"models.{model_config['architecture']}"
    )

    ModelClass = getattr(
        module,
        "Model"
    )

    model = ModelClass(
        model_config
    ).to(device)

    return model

# -----------------------------
# Запуск
# -----------------------------

def main(args: argparse.Namespace):

    global device
    global model

    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Используем устройство: {device}"
    )

    model_config = config["model_config"]

    model = get_model(
        model_config,
        device
    )

    model_path = (
        args.weights
        if args.weights
        else config.get("model_path")
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    logger.info(
        f"Модель загружена из: {model_path}"
    )

    port = config.get(
        "port",
        5000
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ASVspoof detection system"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./config/RawNet2_baseline.conf"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="./models/weights/RawNet2_baseline.pth"
    )

    args = parser.parse_args()

    main(args)