from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
import tempfile
import json
import os

app = FastAPI()

_metadata = dict()

@app.post("/metadata/")
async def extract_metadata(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["./exiftool/exiftool", "-j", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": result.stderr}
            )

        metadata = json.loads(result.stdout)[0] if result.stdout else {}

        for i in metadata.keys():
            if i not in _metadata:
                _metadata[i] = 1
            else: 
                _metadata[i] += 1            

            
        return JSONResponse(content=metadata)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/get-metadata")
def get_metadata():
    return sorted(_metadata.items(), key=lambda x: x[1])

