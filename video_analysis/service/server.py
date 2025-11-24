from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from infer import load_session, infer_image_bytes, infer_video_bytes, detect_mime
app = FastAPI(title="Deepfake Detector API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = None
def get_session():
    global _session
    if _session is None:
        _session = load_session()
    return _session
@app.get("/health")
async def health():
    return {"status": "ok"}
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        kind = detect_mime(file.filename, file.content_type)
        sess = get_session()
        if kind == "video":
            p_ai = infer_video_bytes(sess, content)
        else:
            p_ai = infer_image_bytes(sess, content)
        is_ai = bool(p_ai > 0.5)
        return JSONResponse(content={
            "is_AI": is_ai,
            "probability_of_AI": float(p_ai)
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
      
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8888, reload=False)
