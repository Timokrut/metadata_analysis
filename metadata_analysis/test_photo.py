import subprocess
import json
import math
import sqlite3
from typing import Any
from datetime import datetime

from TAGS_CONSTANTS import CAMERA_TAGS, GPS_TAGS, SCREENSHOT_TAGS, AI_TAGS, AI_SOFTWARE, RICHNESS_TAGS, IGNORE_TAGS, KNOWN_CAMERAS
# ======================================================
# 1. statistical_score      <- твоя SQLite модель
# 2. camera_score           <- признаки настоящей камеры
# 3. gps_score              <- геоданные
# 4. software_score         <- генераторы / редакторы
# 5. ai_tag_score           <- Prompt, Seed и т.д.
# 6. screenshot_score       <- признаки скриншота
# 7. timestamp_score        <- согласованность дат
# 8. metadata_richness      <- полнота EXIF
# 9. compute_group_ratios   <- распределение EXIF / XMP / IPTC
# ======================================================

def compute_group_ratios(metadata):
    exif = 0
    xmp = 0
    iptc = 0

    for tag in metadata.keys():
        if tag.startswith("EXIF:"):
            exif += 1
        elif tag.startswith("XMP:"):
            xmp += 1
        elif tag.startswith("IPTC:"):
            iptc += 1

    total = exif + xmp + iptc
    if total == 0:
        return {
            "exif_ratio": 0,
            "xmp_ratio": 0,
            "iptc_ratio": 0
        }

    return {
        "exif_ratio": exif / total,
        "xmp_ratio": xmp / total,
        "iptc_ratio": iptc / total
    }

def calculate_metadata_likelihood(metadata, db_path="tags.db") -> tuple[Any, Any]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tags = list(metadata.keys())

    total_score = 0
    contributions = []
    matched = 0
    
    tags = [tag for tag in metadata.keys() if tag not in IGNORE_TAGS]
    
    for tag in tags:
        cur.execute("""
            SELECT Ratio
            FROM tag_frequencies
            WHERE Tag=?
        """, (tag,))

        row = cur.fetchone()

        if not row:
            continue

        ratio = row[0]

        total_score += math.log(ratio)
        matched += 1
        contributions.append((tag, math.log(ratio)))

    conn.close()
    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    if matched == 0:
        return contributions[:20], 0

    return contributions[:20], total_score / matched

def compute_camera_score(metadata):
    tags = set(metadata.keys())
    found = len(tags & CAMERA_TAGS)

    return found / len(CAMERA_TAGS)

def compute_gps_score(metadata):
    tags = set(metadata.keys())
    found = len(tags & GPS_TAGS)

    return found / len(GPS_TAGS)

def compute_screenshot_score(metadata):
    tags = set(metadata.keys())
    found = len(tags & SCREENSHOT_TAGS)

    return found / len(SCREENSHOT_TAGS)

def compute_ai_tag_score(metadata):
    tags = set(metadata.keys())
    found = len(tags & AI_TAGS)

    return min(1.0, found / 2)

def compute_software_score(metadata):
    blob = " ".join(
        str(v).lower()
        for v in metadata.values()
    )

    hits = sum(
        1
        for item in AI_SOFTWARE
        if item in blob
    )

    return min(1.0, hits / 2)

def compute_metadata_richness(metadata):
    tags = set(metadata.keys())

    return len(tags & RICHNESS_TAGS) / len(RICHNESS_TAGS)  

def compute_timestamp_score(metadata):
    create = metadata.get("CreateDate")
    modify = metadata.get("ModifyDate")

    if not create or not modify:
        return 0.5

    try:
        c = datetime.strptime(
            create[:19],
            "%Y:%m:%d %H:%M:%S"
        )

        m = datetime.strptime(
            modify[:19],
            "%Y:%m:%d %H:%M:%S"
        )

        if m < c:
            return 0

        return 1

    except Exception:
        return 0.5  

def compute_known_camera_score(metadata:dict):
    camera = 1 if metadata.get("Make", "") in KNOWN_CAMERAS else 0
    model = 1 if metadata.get("Model", "") != "" else 0
    lens_model = 1 if metadata.get("LensModel", "") != "" else 0
    lens_info = 1 if metadata.get("LensInfo", "") != "" else 0

    return (camera + model + lens_model + lens_info) / 4

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

def get_extended_metadata(path):
    result = subprocess.run(  # noqa: UP022
        ["exiftool/exiftool", "-G", "-j", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        data = json.loads(result.stdout)[0]
    except Exception:
        data = {}
    return data

def normalize_statistical_score(score):
    return 1 / (1 + math.exp(-score))

def analyze(path):
    metadata = get_metadata(path)
    metadata_ext = get_extended_metadata(path)
    _, statistical_score = calculate_metadata_likelihood(metadata)
    camera_score = compute_camera_score(metadata)
    gps_score = compute_gps_score(metadata)
    screenshot_score = compute_screenshot_score(metadata)
    software_score = compute_software_score(metadata)
    ai_tag_score = compute_ai_tag_score(metadata)
    metadata_richness = compute_metadata_richness(metadata)
    timestamp_score = compute_timestamp_score(metadata)
    group_ratios = compute_group_ratios(metadata_ext)
    known_camera_score = compute_known_camera_score(metadata)
    
    statistical_score_norm = normalize_statistical_score(statistical_score)

    photo_probability = (
        statistical_score_norm * 0.4 +
        camera_score * 0.25 +
        known_camera_score * 0.2 + 
        gps_score * 0.15 +
        metadata_richness * 0.1 +
        timestamp_score * 0.1
    )

    screenshot_probability = (
        screenshot_score * 0.8 +
        (1 - camera_score) * 0.2
    )

    ai_probability = (
        ai_tag_score * 0.5 +
        software_score * 0.3 +
        (1 - metadata_richness) * 0.1 +
        (1 - camera_score) * 0.1
    )

    total = (
        photo_probability +
        screenshot_probability +
        ai_probability
    )

    photo_probability /= total
    screenshot_probability /= total
    ai_probability /= total
    
    if photo_probability > 0.5:
        verdict = "PHOTO"

    elif screenshot_probability > 0.5:
        verdict = "SCREENSHOT"

    elif ai_probability > 0.5:
        verdict = "AI"

    else:
        verdict = "UNKNOWN"
    
    return {
        "statistical_score": statistical_score,
        "statistical_score_norm": statistical_score_norm,
        "camera_score": camera_score,
        "known_camera_score": known_camera_score,
        "gps_score": gps_score,
        "screenshot_score": screenshot_score,
        "software_score": software_score,
        "ai_tag_score": ai_tag_score,
        "metadata_richness": metadata_richness,
        "timestamp_score": timestamp_score, 
        "exif_xmp_iptc": group_ratios,
        "photo_probability": photo_probability,
        "screenshot_probability": screenshot_probability,
        "ai_probability": ai_probability,
        "verdict": verdict,
        "confidence": max(photo_probability, ai_probability, screenshot_probability)

    }

if __name__ == "__main__":
    paths = ["./iphone.HEIC"]
    names = ["vid"]
    
    for i in range(len(paths)):
        result = analyze(paths[i])
        print(names[i])
        print(json.dumps(
            result,
            indent=2
        ))
        print()
