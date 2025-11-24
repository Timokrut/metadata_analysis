import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_learning_curves(histories, out_path):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for name,h in histories.items():
        plt.plot(h["epoch"], h["train_loss"], label=f"{name} train_loss")
        plt.plot(h["epoch"], h["val_loss"],   label=f"{name} val_loss", linestyle="--")
    plt.title("Learning Curves: Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    for name,h in histories.items():
        plt.plot(h["epoch"], h["train_auc"], label=f"{name} train_auc")
        plt.plot(h["epoch"], h["val_auc"],   label=f"{name} val_auc", linestyle="--")
    plt.title("Learning Curves: AUC"); plt.xlabel("Epoch"); plt.ylabel("ROC AUC"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_roc_pr(results, out_prefix):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for name, r in results.items():
        y_true = r["test"]["y_true"]; y_pred = r["test"]["y_pred"]
        fpr, tpr, _ = roc_curve(y_true, y_pred); auc_val = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],"k--", alpha=0.5)
    plt.title("ROC Curves"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    for name, r in results.items():
        y_true = r["test"]["y_true"]; y_pred = r["test"]["y_pred"]
        prec, rec, _ = precision_recall_curve(y_true, y_pred); ap_val = average_precision_score(y_true, y_pred)
        plt.plot(rec, prec, label=f"{name} (AP={ap_val:.3f})")
    plt.title("Precision-Recall Curves"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_prefix + "_roc_pr.png", dpi=150); plt.close()

def plot_confusions(results, out_prefix):
    n = len(results); plt.figure(figsize=(5*n,4))
    for i,(name,r) in enumerate(results.items()):
        cm = r["test"]["cm"]
        plt.subplot(1,n,i+1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["real","fake"], yticklabels=["real","fake"])
        plt.title(f"{name} Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_prefix + "_cm.png", dpi=150); plt.close()

def plot_calibration(results, out_prefix, bins=10):
    n = len(results); plt.figure(figsize=(5*n,4))
    for i,(name,r) in enumerate(results.items()):
        y_true = r["test"]["y_true"]; y_pred = r["test"]["y_pred"]
        bin_ids = np.floor(y_pred * bins).astype(int); bin_ids = np.clip(bin_ids, 0, bins-1)
        accs, confs = [], []
        for b in range(bins):
            m = bin_ids == b
            if m.sum() == 0:
                accs.append(np.nan); confs.append((b+0.5)/bins); continue
            accs.append((y_true[m] == (y_pred[m]>=0.5)).mean())
            confs.append(y_pred[m].mean())
        plt.subplot(1,n,i+1)
        plt.plot([0,1],[0,1],'k--', alpha=0.5); plt.plot(confs, accs, marker='o', label="Reliability")
        import numpy as np as _np
        ece = np.nanmean(_np.abs(_np.array(accs) - _np.array(confs)))
        plt.title(f"{name} Calibration"); plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_prefix + "_calibration.png", dpi=150); plt.close()

def plot_score_hist(results, out_prefix):
    n = len(results); plt.figure(figsize=(5*n,4))
    for i,(name,r) in enumerate(results.items()):
        y_true = r["test"]["y_true"]; y_pred = r["test"]["y_pred"]
        plt.subplot(1,n,i+1)
        plt.hist(y_pred[y_true==0], bins=30, alpha=0.6, label="real", color="#2ca02c")
        plt.hist(y_pred[y_true==1], bins=30, alpha=0.6, label="fake", color="#d62728")
        plt.title(f"{name} Score distributions"); plt.xlabel("p(fake)"); plt.ylabel("count"); plt.legend()
    plt.tight_layout(); plt.savefig(out_prefix + "_score_hist.png", dpi=150); plt.close()

def build_infographic(histories, results, out_path, cfg):
    figs_dir = os.path.join(cfg["paths"]["out_dir"], "figs"); ensure_dir(figs_dir)
    lc_path = os.path.join(figs_dir, "learning_curves.png")
    rocpr_prefix = os.path.join(figs_dir, "curves")
    plot_learning_curves(histories, lc_path)
    plot_roc_pr(results, rocpr_prefix)
    plot_confusions(results, os.path.join(figs_dir, "confusion"))
    plot_calibration(results, os.path.join(figs_dir, "calib"))
    plot_score_hist(results, os.path.join(figs_dir, "hist"))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,20))
    img1 = plt.imread(lc_path); plt.subplot(3,2,1); plt.imshow(img1); plt.axis('off'); plt.title("Learning Curves")
    img2 = plt.imread(rocpr_prefix + "_roc_pr.png"); plt.subplot(3,2,2); plt.imshow(img2); plt.axis('off'); plt.title("ROC & PR Curves")
    img3 = plt.imread(os.path.join(figs_dir, "confusion_cm.png")); plt.subplot(3,2,3); plt.imshow(img3); plt.axis('off'); plt.title("Confusion Matrices")
    img4 = plt.imread(os.path.join(figs_dir, "calib_calibration.png")); plt.subplot(3,2,4); plt.imshow(img4); plt.axis('off'); plt.title("Calibration Plots")
    img5 = plt.imread(os.path.join(figs_dir, "hist_score_hist.png")); plt.subplot(3,2,5); plt.imshow(img5); plt.axis('off'); plt.title("Score Distributions")
    plt.subplot(3,2,6); plt.axis('off'); plt.title("Summary Metrics")
    rows = []
    for name, r in results.items():
        te = r["test"]; rows.append([name, f"{te['auc']:.4f}", f"{te['ap']:.4f}", f"{te['acc']:.4f}"])
    col_labels = ["Model","ROC AUC","AP","Accuracy"]
    tbl = plt.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center'); tbl.scale(1,2)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
