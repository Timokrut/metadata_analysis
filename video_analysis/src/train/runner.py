import os, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.data.ffpp import build_image_lists
from src.data.datasets import ImageBinaryDataset
from src.model.meso import Meso4, MesoInception4
from src.train.loops import train_model, evaluate_epoch
from sklearn.utils import shuffle as sk_shuffle

def _device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _pos_weight(labels, device):
    p = max(1.0, (len(labels) - sum(labels)) / (sum(labels) + 1e-6))
    return torch.tensor([p], device=device, dtype=torch.float32)

def _build_loaders(cfg):
    D,I,S,P,T = cfg["dataset"], cfg["input"], cfg["split"], cfg["paths"], cfg["train"]
    images_root = D["crops_cache_dir"] if D["data_mode"]=="videos" else D["dataset_root"]
    all_paths, all_labels = build_image_lists(images_root)
    assert len(all_paths)>0, "Датасет пуст. Проверьте пути и структуру real/fake"
    # стратифицированный сплит по изображениям
    idx = np.arange(len(all_paths)); labels_np = np.array(all_labels)
    rng = np.random.RandomState(S["random_seed"]); perm = rng.permutation(len(all_paths))
    idx = idx[perm]; labels_np = labels_np[perm]
    n = len(idx); n_test = int(n*S["test_ratio"]); n_val = int(n*S["val_ratio"]); n_train = n - n_test - n_val
    order = np.argsort(labels_np, kind='stable'); idx = idx[order]
    def stratified_split(idx, labels, n_train, n_val, n_test):
        idx0 = idx[labels[idx]==0]; idx1 = idx[labels[idx]==1]
        def split_part(part_idx, n_tr, n_v, n_te): return part_idx[:n_tr], part_idx[n_tr:n_tr+n_v], part_idx[n_tr+n_v:n_tr+n_v+n_te]
        n0, n1 = len(idx0), len(idx1); r0 = n0/(n0+n1+1e-9)
        t0,v0,te0 = split_part(idx0, int(n_train*r0), int(n_val*r0), int(n_test*r0))
        t1,v1,te1 = split_part(idx1, n_train-int(n_train*r0), n_val-int(n_val*r0), n_test-int(n_test*r0))
        return np.concatenate([t0,t1]), np.concatenate([v0,v1]), np.concatenate([te0,te1])
    train_idx, val_idx, test_idx = stratified_split(idx, labels_np, n_train, n_val, n_test)
    def subset(paths, labels, idxs): return [paths[i] for i in idxs], [int(labels[i]) for i in idxs]
    train_paths, train_labels = subset(all_paths, all_labels, train_idx)
    val_paths,   val_labels   = subset(all_paths, all_labels, val_idx)
    test_paths,  test_labels  = subset(all_paths, all_labels, test_idx)
    train_ds = ImageBinaryDataset(train_paths, train_labels, augment=True,  size=I["image_size"])
    val_ds   = ImageBinaryDataset(val_paths,   val_labels,   augment=False, size=I["image_size"])
    test_ds  = ImageBinaryDataset(test_paths,  test_labels,  augment=False, size=I["image_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              prefetch_factor=2 if cfg["train"]["num_workers"]>0 else None, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              prefetch_factor=2 if cfg["train"]["num_workers"]>0 else None)
    test_loader  = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True,
                              prefetch_factor=2 if cfg["train"]["num_workers"]>0 else None)
    return (train_loader, val_loader, test_loader, train_paths, val_paths, test_paths, train_labels, val_labels, test_labels)

def _experiments(cfg):
    return [
        ("Meso4", lambda: Meso4(num_classes=1, img_size=cfg["input"]["image_size"])),
        ("MesoInception4", lambda: MesoInception4(num_classes=1, img_size=cfg["input"]["image_size"]))
    ]

def train_both_models(cfg):
    device = _device(); P = cfg["paths"]; T = cfg["train"]
    os.makedirs(P["out_dir"], exist_ok=True)
    loaders = _build_loaders(cfg)
    train_loader, val_loader, test_loader, train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = loaders
    pos_w = _pos_weight(train_labels, device) if T["use_pos_weight"] else None

    results, histories = {}, {}
    for name, ctor in _experiments(cfg):
        model, best_path, hist = train_model(name, ctor, cfg, train_loader, val_loader, pos_weight=pos_w, device=device)
        histories[name] = hist
        ckpt = torch.load(best_path, map_location=device)
        best_model = ctor().to(device); best_model.load_state_dict(ckpt["model"]); best_model.eval()
        criterion = torch.nn.BCEWithLogitsLoss()
        te = evaluate_epoch(best_model, test_loader, criterion, device, amp=T["amp"])
        results[name] = {"best_path": best_path, "test": te}
        print(f"\n[{name}] Test AUC: {te['auc']:.4f} | Test AP: {te['ap']:.4f} | Test ACC: {te['acc']:.4f}")

    # сохранить preds и метрики
    preds_dir = os.path.join(P["out_dir"], "preds"); os.makedirs(preds_dir, exist_ok=True)
    metrics = []
    for name, r in results.items():
        y_true = r["test"]["y_true"]; y_pred = r["test"]["y_pred"]
        df = pd.DataFrame({"y_true": y_true, "p_fake": y_pred})
        df["pred"] = (df["p_fake"] >= 0.5).astype(int)
        df.to_csv(os.path.join(preds_dir, f"{name}_test_preds.csv"), index=False)
        cm = r["test"]["cm"]; tn, fp, fn, tp = cm.ravel()
        from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
        auc_val = r["test"]["auc"]; ap_val = r["test"]["ap"]; acc_val = r["test"]["acc"]
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)
        brier = brier_score_loss(y_true, y_pred)
        metrics.append({"model": name, "auc": auc_val, "ap": ap_val, "acc": acc_val,
                        "precision": prec, "recall": rec, "f1": f1, "brier": brier,
                        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    metrics_df = pd.DataFrame(metrics)
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(P["out_dir"], f"metrics_{ts}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # session.json
    session = {
        "timestamp": ts, "config": cfg, "metrics_csv": metrics_path,
        "figs_dir": os.path.join(P["out_dir"], "figs"),
        "preds_dir": preds_dir,
        "exports": []
    }
    session_json = os.path.join(P["out_dir"], f"session_{ts}.json")
    with open(session_json, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    return results, histories, ts, metrics_path

def load_best_results(cfg):
    results, histories, ts, metrics_path = train_both_models(cfg)  # если еще не обучено — обучит
    from src.viz.plots import ensure_dir
    figs_dir = os.path.join(cfg["paths"]["out_dir"], "figs"); ensure_dir(figs_dir)
    infographic_path = os.path.join(cfg["paths"]["out_dir"], f"infographic_{ts}.png")
    return results, histories, infographic_path, metrics_path

def load_results_only(cfg):
    # вспомогательно, если уже обучено; тут можно расширить чтением из session.json
    results, histories, ts, _ = train_both_models(cfg)
    return results

def load_best_models_and_test(cfg):
    device = _device()
    results, histories, ts, metrics_path = train_both_models(cfg)
    models = {}
    for name, ctor in _experiments(cfg):
        ckpt = torch.load(results[name]["best_path"], map_location=device)
        m = ctor().to(device); m.load_state_dict(ckpt["model"]); m.eval()
        models[name] = m
    # вернуть test split
    _,_,_,_,_, test_paths, _,_, test_labels = _build_loaders(cfg)
    return models, test_paths, test_labels

def load_best_models(cfg):
    models, _, _ = load_best_models_and_test(cfg)
    return models
