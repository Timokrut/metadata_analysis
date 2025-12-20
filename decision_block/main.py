from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests
import os
import json
import uuid
from pathlib import Path
import shutil

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SHARED_DATA_PATH = os.getenv("SHARED_DATA_PATH", "/shared_data")

SERVICES = {
    "metadata": os.getenv("META_LINK"),
    "video": os.getenv("VIDEO_LINK"),
    "audio": os.getenv("AUDIO_LINK")
}

uploaded_files = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = Path(original_filename).suffix
    
    saved_filename = f"{file_id}{file_extension}"
    file_path = Path(SHARED_DATA_PATH) / saved_filename
    
    file_size = 0
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        file_size = buffer.tell()

    uploaded_files[file_id] = {
        "filename": file.filename,
        "saved_filename": saved_filename,
        "file_path": str(file_path),
        "size": file_size,
        "upload_time": os.path.getctime(file_path)
    }
    
    return JSONResponse ({
        "file_id": file_id,
        "filename": saved_filename,
        "original_name": original_filename,
        "path": str(file_path),
        "size": file_size,
        "message": "File uploaded successfully"
    })

@app.get("/analyze/metadata/{file_id}")
async def analyze_metadata(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    file_info = uploaded_files[file_id]
    file_path = Path(file_info["file_path"])
    
    try:
        response = requests.post(
            f"{SERVICES['metadata']}/analyze",
            data={"file_path": file_path},
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        print(result)
        
        return JSONResponse({
            "service": "metadata",
            "file_id": file_id,
            "result": result,
            "status": "success"
        })
        
    except requests.exceptions.RequestException as e:
        return JSONResponse({
            "service": "metadata",
            "file_id": file_id,
            "error": str(e),
            "status": "error"
        }, status_code=500)


@app.get("/analyze/video/{file_id}")
async def analyze_video(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    file_info = uploaded_files[file_id]
    file_path = Path(file_info["file_path"])
    
    try:
        response = requests.post(
            f"{SERVICES['video']}/analyze",
            data={"file_path": file_path},
        )
        
        response.raise_for_status()
        result = response.json()
        
        return JSONResponse({
            "service": "video",
            "file_id": file_id,
            "result": result,
            "status": "success"
        })
        
    except requests.exceptions.RequestException as e:
        return JSONResponse({
            "service": "video",
            "file_id": file_id,
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/analyze/audio/{file_id}")
async def analyze_audio(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    file_info = uploaded_files[file_id]
    file_path = Path(file_info["file_path"])

    file_path = convert_video_to_wav_ffmpeg(file_path)
    if not file_path:
        return JSONResponse({
            "service": "audio",
            "file_id": file_id,
            "error": "Can't convert file to .wav",
            "status": "error"
        }, status_code=500)
    
    try:
        response = requests.post(
            f"{SERVICES['audio']}/analyze",
            data={"file_path": file_path},
        )
        
        response.raise_for_status()
        result = response.json()
        
        return JSONResponse({
            "service": "audio",
            "file_id": file_id,
            "result": result,
            "status": "success"
        })
        
    except requests.exceptions.RequestException as e:
        return JSONResponse({
            "service": "audio",
            "file_id": file_id,
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/analyze/all/{file_id}")
async def analyze_all(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    results = {}
    
    try:
        metadata_result = await analyze_metadata(file_id)
        results["metadata"] = json.loads(metadata_result.body)
        
        video_result = await analyze_video(file_id)
        results["video"] = json.loads(video_result.body)
        
        audio_result = await analyze_audio(file_id)
        results["audio"] = json.loads(audio_result.body)
        
        probabilities = []
        for service in ["metadata", "video", "audio"]:
            if results[service]["status"] == "success":
                prob = results[service]["result"].get("probability_of_ai")
                probabilities.append(prob)
            else:
                probabilities.append(0.5)
        
        avg_probability = sum(probabilities) / len(probabilities)
        final_decision = "AI" if avg_probability > 0.3 else "NOT AI"
        
        results["final"] = {
            "average_probability": avg_probability,
            "final_decision": final_decision,
            "metadata_probability": probabilities[0],
            "video_probability": probabilities[1],
            "audio_probability": probabilities[2],
        }
        
        return JSONResponse({
            "file_id": file_id,
            "results": results,
            "status": "complete"
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/files/{file_id}")
async def get_file_info(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    return JSONResponse(uploaded_files[file_id])

@app.get("/files")
async def list_files():
    files_list = []
    for file_id, info in uploaded_files.items():
        files_list.append({
            "file_id": file_id,
            "filename": info["filename"],
            "size": info["size"]
        })
    
    return JSONResponse({"files": files_list})

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    if file_id not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    file_info = uploaded_files[file_id]
    file_path = Path(file_info["file_path"])
    
    if file_path.exists():
        file_path.unlink()
    
    del uploaded_files[file_id]
    
    return JSONResponse({
        "status": "deleted",
        "file_id": file_id,
        "message": "File deleted successfully"
    })

import subprocess
def convert_video_to_wav_ffmpeg(video_path, output_path=None):
    # Проверяем наличие ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      capture_output=True, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ошибка: FFmpeg не установлен или не найден в PATH!")
        print("Установите FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    # Проверяем исходный файл
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Файл '{video_path}' не найден!")
        return False
    
    # Определяем выходной файл
    if output_path is None:
        output_path = video_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path / video_path.with_suffix('.wav').name
    
    # Команда FFmpeg для конвертации
    cmd = [
        'ffmpeg',
        '-i', str(video_path),      # Входной файл
        '-vn',                       # Без видео
        '-acodec', 'pcm_s16le',     # Кодек для WAV
        '-ar', '44100',             # Частота дискретизации
        '-ac', '2',                 # Стерео
        '-y',                       # Перезаписать если файл существует
        str(output_path)
    ]
    
    try:
        print(f"🔄 Конвертируем {video_path.name} в WAV...")
        
        # Запускаем конвертацию
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              check=True)
        
        # Проверяем результат
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ Успешно создан: {output_path}")
            return output_path 
        else:
            print("❌ Ошибка: Выходной файл не создан или пустой")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка FFmpeg: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Неизвестная ошибка: {e}")
        return False
