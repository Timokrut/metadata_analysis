from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from collections import Counter
import subprocess
import json
import sqlite3
import tempfile
import shutil
import sqlite3 

app = FastAPI()

templates = Jinja2Templates(directory="templates")

def get_metadata(path):
    """Extract EXIF metadata from a file using exiftool."""
    result = subprocess.run(  # noqa: UP022
        ["exiftool/exiftool", "-j", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        data = json.loads(result.stdout)[0]
    except Exception:
        data = {}
    return data

def analyze_tag_frequencies_sql(dataset_dir_real, dataset_dir_ai, db_path="tags.db"):
    def collect_tags(path_dir):
        tags = []
        for f in Path(path_dir).glob("**/*.*"):
            md = get_metadata(f)
            tags.extend(md.keys())
        return tags

    real_tags = collect_tags(dataset_dir_real)
    ai_tags = collect_tags(dataset_dir_ai)

    real_counts = Counter(real_tags)
    ai_counts = Counter(ai_tags)

    all_tags = set(real_counts.keys()) | set(ai_counts.keys())
    data_to_insert = []
    for tag in all_tags:
        rf = real_counts[tag]
        af = ai_counts[tag]
        diff = rf - af
        ratio = (rf + 1) / (af + 1)
        data_to_insert.append((tag, rf, af, diff, ratio))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tag_frequencies (
            Tag TEXT PRIMARY KEY,
            RealFreq INTEGER,
            AIFreq INTEGER,
            Diff INTEGER,
            Ratio REAL
        )
    """)
    cur.executemany("""
        INSERT OR REPLACE INTO tag_frequencies (Tag, RealFreq, AIFreq, Diff, Ratio)
        VALUES (?, ?, ?, ?, ?)
    """, data_to_insert)
    conn.commit()
    conn.close()
    print(f"[INFO] Saved {len(data_to_insert)} tags to {db_path}")

def get_top_tags(db_path="tags.db", top_n=10):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT Tag FROM tag_frequencies
        ORDER BY Ratio DESC
        LIMIT ?
    """, (top_n,))
    top_tags = [row[0] for row in cur.fetchall()]
    conn.close()
    return top_tags

def is_ai_statistical_sql(metadata, db_path="tags.db", top_n=10):
    top_tags = get_top_tags(db_path, top_n)
    present_tags = set(metadata.keys())
    overlap = present_tags & set(top_tags)
    score = len(overlap) / len(top_tags)

    is_ai = score < 0.3
    return is_ai, f"Tag overlap score={score:.2f}"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    try:
        metadata = get_metadata(tmp_path)
        if not metadata:
            raise HTTPException(status_code=400, detail="No EXIF metadata found.")

        is_ai, explanation = is_ai_statistical_sql(metadata)
        preview_metadata = dict(list(metadata.items())[:10])

        result = {
            "filename": file.filename,
            "is_ai": is_ai,
            "explanation": explanation,
            "metadata": preview_metadata
        }

        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    finally:
        tmp_path.unlink(missing_ok=True)

### FOR TESTING PURPOSES ###
from random import random

@app.get("/make-decision")
def analyze_metadata():
    return {"probability_of_ai": random()}
