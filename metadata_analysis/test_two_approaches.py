import json
import subprocess
from collections import Counter
from pathlib import Path
import sqlite3

def get_metadata(path):
    """Extract EXIF metadata from a file using exiftool."""
    result = subprocess.run(  # noqa: UP022
        ["exiftool", "-j", str(path)],
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

def get_top_tags(db_path="tags.db", top_n=100):
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

def is_ai_statistical_sql(metadata, db_path="tags.db", top_n=100):
    top_tags = get_top_tags(db_path, top_n)
    present_tags = set(metadata.keys())
    overlap = present_tags & set(top_tags)
    score = len(overlap) / len(top_tags)

    is_ai = score < 0.3
    return is_ai, f"Tag overlap score={score:.2f}"

def validate_dataset(validate_dir, db_path="tags.db", results_table="validation_results"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {results_table} (
            FilePath TEXT PRIMARY KEY,
            IsAI INTEGER,
            Score REAL,
            Comment TEXT
        )
    """)

    # получаем список файлов
    files = list(Path(validate_dir).glob("**/*.*"))
    print(f"[INFO] Found {len(files)} files to check")

    data_to_insert = []

    for f in files:
        md = get_metadata(f)
        is_ai, comment = is_ai_statistical_sql(md, db_path=db_path, top_n=10)
        score = float(comment.split("=")[-1]) 

        data_to_insert.append((str(f), int(is_ai), score, comment))

    cur.executemany(f"""
        INSERT OR REPLACE INTO {results_table} (FilePath, IsAI, Score, Comment)
        VALUES (?, ?, ?, ?)
    """, data_to_insert)

    conn.commit()
    conn.close()
    print(f"[INFO] Saved {len(data_to_insert)} results to table {results_table}")


if __name__ == "__main__":
    validate_dataset("/home/user/Coding/metadata_analysis/dataset/images")
