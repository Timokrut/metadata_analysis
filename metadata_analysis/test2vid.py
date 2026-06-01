import re
from typing import Dict, Any, List
import subprocess
import json
import math
import sqlite3
from typing import Any
from datetime import datetime

# ------------------------------------------------------------
# Ключевые слова для поиска в именах тегов (без привязки к группе)
# ------------------------------------------------------------
CAMERA_TAG_PATTERNS = [
    r'\bmake\b', r'\bmodel\b', r'\blensmodel\b', r'\blensinfo\b',
    r'\biso\b', r'\bexposuretime\b', r'\bfnumber\b', r'\baperture\b',
    r'\bfocallength\b', r'\bflash\b', r'\bwhitebalance\b',
    r'\bmeteringmode\b', r'\bexposuremode\b'
]

GPS_TAG_PATTERNS = [
    r'\bgpslatitude\b', r'\bgpslongitude\b', r'\bgpsaltitude\b', r'\bgpsdatestamp\b'
]

IGNORE_TAGS = {
    "SourceFile",
    "FileName",
    "Directory",
    "FileSize",
    "FileModifyDate",
    "FileAccessDate",
    "FileInodeChangeDate",
    "FilePermissions",
    "ExifToolVersion"
}

RICHNESS_TAG_PATTERNS = [
    r'\bduration\b', r'\btrackduration\b', r'\bmediaduration\b',
    r'\bvideoframerate\b', r'\bavgbitrate\b', r'\bimagewidth\b', r'\bimageheight\b',
    r'\bcompressorid\b', r'\bencodername\b',
    r'\bmake\b', r'\bmodel\b', r'\blensmodel\b',
    r'\bcreatetime\b', r'\bcreationdate\b', r'\bmodifydate\b',
    r'\bgpslatitude\b', r'\bgpslongitude\b'
]

KNOWN_CAMERAS = {
    "apple", "samsung", "xiaomi", "canon", "sony",
    "nikon", "fuji", "panasonic", "dji", "gopro"
}

AI_VIDEO_SOFTWARE = {
    "runway", "gen-2", "gen-3", "pika", "pika labs",
    "sora", "openai", "stable video diffusion", "svd",
    "modelscope", "zeroscope", "vidgen", "moonvalley",
    "kling", "hailuo", "luma"
}

# ------------------------------------------------------------
# Новые функции, работающие с расширенными метаданными (dict из exiftool -G -j)
# ------------------------------------------------------------
def _tag_has_pattern(extended_key: str, patterns: List[str]) -> set:
    """Возвращает множество паттернов, найденных в последней части ключа (после ':')."""
    tag_name = extended_key.split(':')[-1].lower()
    return {p for p in patterns if re.search(p, tag_name)}

def compute_camera_score(metadata_ext: Dict) -> float:
    found = set()
    for key in metadata_ext:
        found.update(_tag_has_pattern(key, CAMERA_TAG_PATTERNS))
    return len(found) / len(CAMERA_TAG_PATTERNS)

def compute_known_camera_score(metadata_ext: Dict) -> float:
    make_found = False
    model_found = False
    lens_model_found = False
    lens_info_found = False

    for key, val in metadata_ext.items():
        tag_last = key.split(':')[-1].lower()
        val_str = str(val).lower()
        if tag_last == 'make' and any(m in val_str for m in KNOWN_CAMERAS):
            make_found = True
        elif tag_last == 'model' and val:
            model_found = True
        elif tag_last == 'lensmodel' and val:
            lens_model_found = True
        elif tag_last == 'lensinfo' and val:
            lens_info_found = True
    return sum([make_found, model_found, lens_model_found, lens_info_found]) / 4

def compute_gps_score(metadata_ext: Dict) -> float:
    found = set()
    for key in metadata_ext:
        found.update(_tag_has_pattern(key, GPS_TAG_PATTERNS))
    return len(found) / len(GPS_TAG_PATTERNS)

def compute_metadata_richness(metadata_ext: Dict) -> float:
    found = set()
    for key in metadata_ext:
        found.update(_tag_has_pattern(key, RICHNESS_TAG_PATTERNS))
    return len(found) / len(RICHNESS_TAG_PATTERNS)

def compute_ai_software_score(metadata_ext: Dict) -> float:
    blob = " ".join(str(v).lower() for v in metadata_ext.values())
    hits = sum(1 for kw in AI_VIDEO_SOFTWARE if kw in blob)
    return min(1.0, hits / 2)

def compute_timestamp_score(metadata_ext: Dict) -> float:
    # Ищем даты создания/модификации в любых группах
    create_vals = []
    modify_vals = []
    for key, val in metadata_ext.items():
        tag_low = key.split(':')[-1].lower()
        if 'createdate' in tag_low or 'creationdate' in tag_low:
            create_vals.append(str(val))
        if 'modifydate' in tag_low:
            modify_vals.append(str(val))
    if not create_vals or not modify_vals:
        return 0.5
    try:
        c = datetime.strptime(create_vals[0][:19], "%Y:%m:%d %H:%M:%S")
        m = datetime.strptime(modify_vals[0][:19], "%Y:%m:%d %H:%M:%S")
        return 1.0 if m >= c else 0.0
    except Exception:
        return 0.5

def calculate_metadata_likelihood(metadata, db_path="video_tags.db") -> tuple[Any, Any]:
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

def get_metadata(path):
    result = subprocess.run(
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
    result = subprocess.run(
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

# ------------------------------------------------------------
# Модифицированная analyze() – убраны скриншоты, используются новые метрики
# ------------------------------------------------------------
def analyze(path):
    metadata_ext = get_extended_metadata(path)   # словарь с группами
    metadata_flat = get_metadata(path)
    print(metadata_ext)
    _, statistical_score = calculate_metadata_likelihood(metadata_flat)
    statistical_score_norm = normalize_statistical_score(statistical_score)

    camera_score = compute_camera_score(metadata_ext)
    known_camera_score = compute_known_camera_score(metadata_ext)
    gps_score = compute_gps_score(metadata_ext)
    ai_sw_score = compute_ai_software_score(metadata_ext)
    richness = compute_metadata_richness(metadata_ext)
    timestamp_score = compute_timestamp_score(metadata_ext)

    # Вероятности двух классов
    real_prob = (statistical_score_norm * 0.35 +
                 camera_score * 0.25 +
                 known_camera_score * 0.15 +
                 gps_score * 0.1 +
                 richness * 0.1 +
                 timestamp_score * 0.05)

    ai_prob = (ai_sw_score * 0.5 +
               (1 - richness) * 0.2 +
               (1 - camera_score) * 0.15 +
               (1 - statistical_score_norm) * 0.15)

    total = real_prob + ai_prob
    if total > 0:
        real_prob /= total
        ai_prob /= total
    else:
        real_prob = ai_prob = 0.5

    if real_prob > 0.6:
        verdict = "REAL"
    elif ai_prob > 0.6:
        verdict = "AI"
    else:
        verdict = "UNKNOWN"

    return {
        "statistical_score": statistical_score,
        "statistical_score_norm": statistical_score_norm,
        "camera_score": camera_score,
        "known_camera_score": known_camera_score,
        "gps_score": gps_score,
        "ai_software_score": ai_sw_score,
        "metadata_richness": richness,
        "timestamp_score": timestamp_score,
        "real_probability": real_prob,
        "ai_probability": ai_prob,
        "verdict": verdict,
        "confidence": max(real_prob, ai_prob)
    }


result = analyze("./ai/ai (28).mp4")
print(json.dumps(
    result,
    indent=2
))