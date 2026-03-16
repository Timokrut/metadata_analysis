import json
import os
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import shutil

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

SHARED_DATA_PATH = Path(os.getenv("SHARED_DATA_PATH", "/shared_data"))

TOP_N_TAGS = 10

CAMERA_TAGS = {
    "Make", 
    "Model", 
    "Lens", 
    "ISO", 
    "ExposureTime", 
    "FNumber"
}

SOFTWARE_TAGS = {
    "Software",
    "Creator",
    "Generator",
    "Model",
    "StableDiffusion",
    "Midjourney",
    "Runway"
}

GPS_TAGS = {
    "GPSLatitude", 
    "GPSLongitude"
}

TIMESTAMP_TAGS = {
    "CreateDate", 
    "ModifyDate"
}

def compute_avg_tag_counts(dataset_dir_real, dataset_dir_ai):
    def collect_counts(path_dir):
        counts = []
        for f in Path(path_dir).glob("**/*.*"):
            md = get_metadata(f)
            if md:
                counts.append(len(md))
        return counts

    real_counts = collect_counts(dataset_dir_real)
    ai_counts = collect_counts(dataset_dir_ai)

    avg_real = sum(real_counts) / len(real_counts) if real_counts else 0
    avg_ai = sum(ai_counts) / len(ai_counts) if ai_counts else 0

    return avg_real, avg_ai

def get_top_ai_tags(db_path="tags.db", top_n=TOP_N_TAGS):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT Tag FROM tag_frequencies
        ORDER BY Ratio ASC
        LIMIT ?
    """, (top_n,))

    tags = [row[0] for row in cur.fetchall()]
    conn.close()

    return tags

def analyze_special_metadata(metadata):
    tags = set(metadata.keys())

    has_camera_metadata = any(tag in tags for tag in CAMERA_TAGS)
    software_flag = any(tag in tags for tag in SOFTWARE_TAGS)
    gps_present = all(tag in tags for tag in GPS_TAGS)

    timestamp_anomaly = False
    for tag in TIMESTAMP_TAGS:
        if tag in metadata:
            try:
                year = int(str(metadata[tag])[:4])
                if year < 2000:
                    timestamp_anomaly = True

            except:
                pass

    return {
        "has_camera_metadata": has_camera_metadata,
        "software_flag": software_flag,
        "gps_present": gps_present,
        "timestamp_anomaly": timestamp_anomaly
    }

def get_metadata(path):
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

def get_top_tags(db_path="tags.db", top_n=TOP_N_TAGS):
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

def is_ai_statistical_sql(metadata, db_path="tags.db", top_n=TOP_N_TAGS):
    top_tags = get_top_tags(db_path, top_n)
    present_tags = set(metadata.keys())
    overlap = present_tags & set(top_tags)
    score = len(overlap) / len(top_tags)

    is_ai = score < 0.3
    return is_ai, f"Tag overlap score={score:.2f}"


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        temp_file_path = f"/tmp/{file.filename}"

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        metadata = get_metadata(temp_file_path)
        print(metadata)
        if not metadata:
            raise HTTPException(status_code=400, detail="No EXIF metadata found.")

        # is_ai, explanation = is_ai_statistical_sql(metadata)
        preview_metadata = dict(list(metadata.items())[:10])

        analysis = analyze_metadata_extended(metadata)

        result = {
            "filename": temp_file_path,
            "analysis": analysis,
            "metadata_preview": preview_metadata 
        }

        print(f"\n\n{result}")
        return result

    except Exception as e:
        print(e)
        return {"status": "failed", "data": e}

def analyze_metadata_extended(metadata, db_path="tags.db", top_n=TOP_N_TAGS):
    present_tags = set(metadata.keys())

    real_top_tags = get_top_tags(db_path, top_n)
    ai_top_tags = get_top_ai_tags(db_path, top_n)

    print("\n\n\n\n")
    print(real_top_tags)
    print("\n\n\n\n")
    print(ai_top_tags)
    print("\n\n\n\n")
    print(present_tags)
    print("\n\n\n\n")

    real_overlap = present_tags & set(real_top_tags)
    ai_overlap = present_tags & set(ai_top_tags)

    real_overlap_score = len(real_overlap) / len(real_top_tags)
    ai_overlap_score = len(ai_overlap) / len(ai_top_tags)

    tag_count = len(metadata)

    special_metrics = analyze_special_metadata(metadata)

    is_ai = real_overlap_score < 0.3

    return {
        "is_ai": is_ai,

        "scores": {
            "tag_count": tag_count,
            "real_overlap_score": real_overlap_score,
            "ai_overlap_score": ai_overlap_score
        },

        "top_tags_current_file": list(present_tags)[:10],

        "overlap_tags": {
            "real_overlap_tags": list(real_overlap),
            "ai_overlap_tags": list(ai_overlap)
        },

        "special_metrics": special_metrics
    }