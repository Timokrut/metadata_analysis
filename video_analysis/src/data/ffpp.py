import os, re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from facenet_pytorch import MTCNN

def _cfg(cfg):  # удобные алиасы
    D = cfg["dataset"]; S = cfg["split"]; I = cfg["input"]; P = cfg["paths"]; T = cfg["train"]
    return D,S,I,P,T

def download_ffpp_kagglehub(kaggle_id: str, dst: Path) -> str:
    import kagglehub
    path = kagglehub.dataset_download(kaggle_id)
    candidate = Path(path) / "FaceForensics++_C23"
    root = str(candidate.resolve()) if candidate.exists() else str(Path(path).resolve())
    out_root = dst / "FaceForensics++_C23"
    out_root.parent.mkdir(parents=True, exist_ok=True)
    # символическая ссылка/копия
    if not out_root.exists():
        try:
            out_root.symlink_to(Path(root), target_is_directory=True)
        except Exception:
            pass
    return root

def discover_ffpp(cfg: dict) -> pd.DataFrame:
    D,_,_,_,_ = _cfg(cfg)
    root = D["dataset_root"]
    assert os.path.isdir(root), f"Not found: {root}"
    all_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    skip_dirs = {"csv", ".ipynb_checkpoints"}
    data_dirs = [d for d in all_dirs if d not in skip_dirs]
    if D.get("manipulations_include") is None:
        fake_dirs = [d for d in data_dirs if d.lower() != D["original_dir"].lower()]
    else:
        fake_dirs = D["manipulations_include"]

    rows = []
    for d in data_dirs:
        d_path = os.path.join(root, d)
        vids = []
        for ext in D["video_exts"]:
            vids.extend(glob.glob(os.path.join(d_path, f"**/*{ext}"), recursive=True))
        label = 0 if d.lower() == D["original_dir"].lower() else (1 if d in fake_dirs else None)
        if label is None: 
            continue
        for p in vids:
            rows.append({
                "path": os.path.abspath(p),
                "label": int(label),
                "manipulation": d,
                "video_id": os.path.splitext(os.path.basename(p))[0],
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df

def _try_read_splits_from_csv(csv_dir: str) -> pd.DataFrame | None:
    if not os.path.isdir(csv_dir): return None
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    if len(csv_files)==0: return None
    # вариант 1: единственный csv со столбцом split
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            cols = [c.lower() for c in df.columns]
        except Exception:
            continue
        if "split" in cols and ("video_id" in cols or "filename" in cols):
            vid_col = "video_id" if "video_id" in df.columns else "filename"
            out = df[[vid_col, "split"]].copy()
            out.columns = ["video_id", "split"]
            out["split"] = out["split"].str.lower().map({"train":"train","val":"val","valid":"val","dev":"val","test":"test"})
            return out.dropna()
    # вариант 2: train/val/test csv
    split_rows = []
    for name in ["train","val","test"]:
        cand = [fp for fp in csv_files if re.search(fr"{name}\.csv$", os.path.basename(fp), re.I)]
        if not cand: continue
        try:
            df = pd.read_csv(cand[0])
            cols = [c.lower() for c in df.columns]
            vid_col = "video_id" if "video_id" in df.columns else ("filename" if "filename" in cols else None)
            if vid_col is None: continue
            tmp = df[[vid_col]].copy(); tmp.columns = ["video_id"]; tmp["split"] = name
            split_rows.append(tmp)
        except Exception:
            pass
    if len(split_rows):
        return pd.concat(split_rows, axis=0).reset_index(drop=True)
    return None

def make_splits(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    D,S,_,_,_ = _cfg(cfg)
    csv_dir = os.path.join(D["dataset_root"], "csv")
    if S["use_csv_splits"]:
        map_df = _try_read_splits_from_csv(csv_dir)
        if map_df is not None and len(map_df):
            merged = df.merge(map_df, on="video_id", how="left")
            merged["split"] = merged["split"].fillna("train")
            return merged
    # если нет csv-сплитов — стратифицируем 80/10/10 по видео
    from sklearn.model_selection import StratifiedShuffleSplit
    rng = np.random.RandomState(S["random_seed"])
    df_shuf = df.sample(frac=1.0, random_state=S["random_seed"]).reset_index(drop=True)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=S["test_ratio"], random_state=S["random_seed"])
    idx_train_val, idx_test = next(sss1.split(df_shuf["video_id"], df_shuf["label"]))
    tv = df_shuf.iloc[idx_train_val].reset_index(drop=True)
    test = df_shuf.iloc[idx_test].reset_index(drop=True)
    val_size = S["val_ratio"] / (1.0 - S["test_ratio"])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=S["random_seed"]+1)
    idx_train, idx_val = next(sss2.split(tv["video_id"], tv["label"]))
    train = tv.iloc[idx_train].copy(); train["split"]="train"
    val   = tv.iloc[idx_val].copy();   val["split"]="val"
    test  = test.copy();               test["split"]="test"
    out = pd.concat([train,val,test], axis=0).reset_index(drop=True)
    return out

def expand_box(box, pad_ratio, W, H):
    x1,y1,x2,y2 = box
    bw, bh = x2-x1, y2-y1
    px, py = bw*pad_ratio, bh*pad_ratio
    nx1 = max(0, int(x1 - px)); ny1 = max(0, int(y1 - py))
    nx2 = min(W-1, int(x2 + px)); ny2 = min(H-1, int(y2 + py))
    return nx1, ny1, nx2, ny2

def extract_ffpp_crops(ffpp_df: pd.DataFrame, cfg: dict, device="cuda"):
    D,S,I,P,T = _cfg(cfg)
    out_root = D["crops_cache_dir"]
    os.makedirs(out_root, exist_ok=True)
    dev = "cuda" if (device=="cuda") else "cpu"
    mtcnn = MTCNN(keep_all=True, device=dev, selection_method='probability', post_process=False)
    rows = ffpp_df.copy()
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Extract faces"):
        vp = row["path"]
        label_name = "real" if int(row["label"])==0 else "fake"
        rel_base = row["video_id"]
        out_dir = os.path.join(out_root, label_name, rel_base)
        os.makedirs(out_dir, exist_ok=True)
        existing = glob.glob(os.path.join(out_dir, "*.jpg"))
        target = D["frames_per_video"] * D["faces_per_frame"]
        if len(existing) >= target:
            continue
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release(); continue
        idxs = np.linspace(0, total-1, num=D["frames_per_video"], dtype=int)
        saved = 0
        for fi, fidx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            boxes, probs = mtcnn.detect(rgb)
            if boxes is None or len(boxes)==0: continue
            order = np.argsort(-np.array(probs))[:D["faces_per_frame"]]
            for k in order:
                bx = boxes[k]
                x1,y1,x2,y2 = expand_box(bx, D["face_pad"], W, H)
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(W, int(x2));  y2 = min(H, int(y2))
                if x2-x1 < D["min_face_size"] or y2-y1 < D["min_face_size"]:
                    continue
                crop = rgb[y1:y2, x1:x2]
                crop = cv2.resize(crop, (I["image_size"], I["image_size"]), interpolation=cv2.INTER_LINEAR)
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, f"f{fi:03d}_k{k}.jpg"), bgr)
                saved += 1
        cap.release()
    print("Crops saved to:", out_root)

def build_image_lists(root: str):
    paths, labels = [], []
    for label_name, y in [("real",0),("fake",1)]:
        for ext in (".jpg",".png",".jpeg",".bmp",".webp"):
            for p in glob.glob(os.path.join(root, label_name, "**", f"*{ext}"), recursive=True):
                paths.append(p); labels.append(y)
    return paths, labels
