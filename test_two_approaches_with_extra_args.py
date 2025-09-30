import json
import subprocess
from pathlib import Path
from collections import Counter
import pandas as pd

def get_metadata(path, extra_args=None):
    if extra_args is None:
        extra_args = ["-j"] # only json 
    
    cmd = ["exiftool"] + extra_args + [str(path)]
    result = subprocess.run(  # noqa: UP022
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        data = json.loads(result.stdout)[0]
    except Exception:
        data = {}
    return data


# --- Approach A: Rule-based heuristic ---
def is_ai_rule_based(metadata):
    suspicious = False
    reasons = []

    # 1. Missing camera info
    if not metadata.get("Make") or not metadata.get("Model"):
        suspicious = True
        reasons.append("Missing camera Make/Model")

    # 2. Suspicious software
    software = metadata.get("Software", "").lower()
    if any(word in software for word in ["stable", "diffusion", "midjourney", "dalle", "ai"]):
        suspicious = True
        reasons.append(f"Suspicious software: {software}")

    # 3. Too few tags
    if len(metadata.keys()) < 5:
        suspicious = True
        reasons.append("Too few metadata tags")

    return suspicious, reasons


# --- Approach B: Statistical tag frequency analysis ---
def analyze_tag_frequencies(dataset_dir_real, dataset_dir_ai, extra_args=None):
    def collect_tags(path_dir):
        tags = []
        for f in Path(path_dir).glob("**/*.*"):
            md = get_metadata(f, extra_args)
            tags.extend(md.keys())
        return tags

    real_tags = collect_tags(dataset_dir_real)
    ai_tags = collect_tags(dataset_dir_ai)

    real_counts = Counter(real_tags)
    ai_counts = Counter(ai_tags)

    all_tags = set(real_counts.keys()) | set(ai_counts.keys())
    data = []
    for tag in all_tags:
        data.append({
            "Tag": tag,
            "RealFreq": real_counts[tag],
            "AIFreq": ai_counts[tag],
        })
    df = pd.DataFrame(data)
    df["Diff"] = df["RealFreq"] - df["AIFreq"]
    df["Ratio"] = (df["RealFreq"] + 1) / (df["AIFreq"] + 1)
    return df.sort_values("Ratio", ascending=False)


def is_ai_statistical(metadata, top_tags):
    present_tags = set(metadata.keys())
    overlap = present_tags & set(top_tags)
    score = len(overlap) / len(top_tags)

    return score < 0.3, f"Tag overlap score={score:.2f}"


if __name__ == "__main__":
    # Run with all metadata
    meta = get_metadata("not_ai_generated.HEIC", extra_args=["-j", "-G", "-all"])
    print(meta)
    print()
    print()
    print()

    # Run with EXIF only
    meta = get_metadata("not_ai_generated.HEIC", extra_args=["-j", "-G", "-EXIF:*"])
    print(meta)
    print()
    print()
    print()

    # Run with IPTC + XMP only
    meta = get_metadata("not_ai_generated.HEIC", extra_args=["-j", "-G", "-IPTC:*", "-XMP:*"])
    print(meta)
    print()
    print()
    print()


    # Compare statistical distributions
    df = analyze_tag_frequencies(
        "dataset/real", 
        "dataset/ai", 
        extra_args=["-j", "-G", "-EXIF:*", "-IPTC:*", "-XMP:*"]
    )
    print(df.head(20))

