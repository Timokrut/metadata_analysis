import json
import subprocess
from collections import Counter
from pathlib import Path
import pandas as pd

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


# --- Approach A: Rule-based ---
def is_ai_rule_based(metadata):
    suspicious = False
    reasons = []

    # 1. Missing camera info
    if not metadata.get("Make") or not metadata.get("Model"):
        suspicious = True
        reasons.append("Missing camera Make/Model")

    # 2. Software field indicates editing/AI
    software = str(metadata.get("Software", "")).lower()
    if any(word in software for word in ["stable", "diffusion", "midjourney", "dalle", "ai"]):
        suspicious = True
        reasons.append(f"Suspicious software: {software}")

    # 3. Very few tags → suspicious
    if len(metadata.keys()) < 5:
        suspicious = True
        reasons.append("Too few metadata tags")

    return suspicious, reasons


# --- Approach B: Statistical tag frequency analysis ---
def analyze_tag_frequencies(dataset_dir_real, dataset_dir_ai):
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

    # Build a DataFrame for comparison
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

    # if too few top tags match → likely AI
    return score < 0.3, f"Tag overlap score={score:.2f}"

if __name__ == "__main__":
    meta = get_metadata("not_ai_generated.HEIC")

    # Rule-based
    flag, reasons = is_ai_rule_based(meta)
    print(f"Rule-based result: \nflag: {'suspicious' if flag else 'not suspicious'}\nwhat is suspicious: {'None' if len(reasons) == 0 else ", ".join(reasons)}")

    # Statistical approach
    df = analyze_tag_frequencies("dataset/real", "dataset/ai")
    print(df.head(10))   # see top distinguishing tags

    # Pick top 25% tags
    top_tags = df["Tag"].head(int(len(df)*0.25)).tolist()

    flag, reason = is_ai_statistical(meta, top_tags)
    print(f"Statistical approach\nSuspicious: {flag}\nReason:{reason}")
 
