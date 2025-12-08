import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import io
import soundfile as sf
import librosa

from torch import Tensor
from importlib import import_module
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

device = None
model = None

app = FastAPI(title="ASVspoof Detection System")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Основной endpoint для загрузки и обработки аудиофайлов
    """
    # Валидация файла
    validate_uploaded_file(file)

    try:
        # Чтение и конвертация аудиофайла
        audio_data, sample_rate = await read_and_convert_audio_file(file)

        # Обработка аудиоданных и получение результата
        result = process_audio_data(audio_data, sample_rate)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудиофайла: {str(e)}")


def validate_uploaded_file(file: UploadFile) -> None:
    """
    Валидация загружаемого файла
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не выбран")

    supported_formats = {'.mp3', '.wav', '.flac', '.aac', '.m4a', '.mp4', '.ogg', '.wma'}
    file_ext = '.' + file.filename.lower().split('.')[-1] if '.' in file.filename else ''

    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f'Неподдерживаемый формат файла. Поддерживаются: {", ".join(supported_formats)}'
        )


async def read_and_convert_audio_file(file: UploadFile) -> Tuple[np.ndarray, int]:
    """
    Чтение и конвертация аудиофайла в унифицированный формат
    """
    contents = await file.read()
    file_ext = '.' + file.filename.lower().split('.')[-1]

    if file_ext == '.flac':
        # Обработка FLAC файлов
        audio_data, sample_rate = read_flac_audio(contents)
    else:
        # Конвертация других форматов в FLAC
        audio_data, sample_rate = convert_to_flac(contents)

    return audio_data, sample_rate


def read_flac_audio(file_contents: bytes) -> Tuple[np.ndarray, int]:
    """
    Чтение FLAC аудио из bytes
    """
    flac_buffer = io.BytesIO(file_contents)
    audio_data, sample_rate = sf.read(flac_buffer)
    return audio_data, sample_rate


def convert_to_flac(file_contents: bytes) -> Tuple[np.ndarray, int]:
    """
    Конвертация различных аудиоформатов в FLAC
    """
    audio_buffer = io.BytesIO(file_contents)
    audio_data, sample_rate = librosa.load(audio_buffer, sr=16000)

    flac_buffer = io.BytesIO()
    sf.write(flac_buffer, audio_data, sample_rate, format='FLAC')
    flac_buffer.seek(0)

    audio_data, sample_rate = sf.read(flac_buffer)
    return audio_data, sample_rate


def process_audio_data(audio_data: np.ndarray, sample_rate: int) -> Dict[str, str]:
    """
    Основная функция обработки аудиоданных через модель
    """
    # Подготовка данных для модели
    batch_x = prepare_audio_for_model(audio_data)

    # Проверка инициализации модели
    check_model_initialization()

    # Получение предсказания от модели
    score, features = get_model_prediction(batch_x)

    # Форматирование результата
    result = format_prediction_result(score, features)

    return result


def prepare_audio_for_model(audio_data: np.ndarray) -> torch.Tensor:
    """
    Подготовка аудиоданных для подачи в модель
    """
    # Паддинг аудиоданных
    audio_padded = pad_audio(audio_data, 64600)
    x_inp = Tensor(audio_padded).ravel()

    # Формирование батча
    if x_inp.size() == torch.Size([129200]):
        batch_x = x_inp.reshape((2, 64600)).to(device)
    else:
        batch_x = x_inp.reshape((1, 64600)).to(device)

    print(f"Подготовленные данные для модели: {batch_x.shape}")
    return batch_x


def pad_audio(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """
    Паддинг аудиосигнала до заданной длины
    """
    x_len = len(x)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(np.ceil(max_len / x_len))
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def check_model_initialization() -> None:
    """
    Проверка инициализации модели и устройства
    """
    if model is None or device is None:
        raise HTTPException(status_code=500, detail="Модель не инициализирована")


def get_model_prediction(batch_x: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    Получение предсказания от модели
    """
    model.eval()
    with torch.no_grad():
        batch_coef, batch_out = model(batch_x)
        batch_score = float(F.softmax(batch_out, dim=-1)[0][1].item())

    print(f"Результат модели: {batch_score}")
    return batch_score, batch_coef


def format_prediction_result(score: float, features: torch.Tensor) -> Dict[str, str]:
    """
    Форматирование результата предсказания
    """
    features_cpu = Tensor(features).to('cpu')

    T_max = str(features_cpu[:, 0:32].numpy().mean())
    T_avg = str(features_cpu[:, 32:64].numpy().mean())
    S_max = str(features_cpu[:, 64:96].numpy().mean())
    S_avg = str(features_cpu[:, 96:128].numpy().mean())

    return {
        "score": score,
        "T_max": T_max,
        "T_avg": T_avg,
        "S_max": S_max,
        "S_avg": S_avg
    }


def get_model(model_config: Dict, device: torch.device) -> torch.nn.Module:
    """
    Загрузка модели по конфигурации
    """
    module = import_module(f"models.{model_config['architecture']}")
    ModelClass = getattr(module, "Model")
    model = ModelClass(model_config).to(device)
    return model


def main(args: argparse.Namespace) -> None:
    """
    Основная функция инициализации и запуска приложения
    """
    global device, model

    # Загрузка конфигурации
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Инициализация модели
    model_config = config["model_config"]
    model = get_model(model_config, device)

    # Загрузка весов модели
    checkpoint = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    print("Модель загружена из:", config["model_path"])

    # Запуск сервера
    port = config.get("port", 5000)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=False,
                        default="./config/AASIST.conf")

    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    args = parser.parse_args()
    main(args)