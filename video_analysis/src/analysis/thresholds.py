import os, numpy as np, matplotlib.pyplot as plt, pandas as pd

def run_threshold_curves(results: dict, cfg: dict):
    out_dir = os.path.join(cfg["paths"]["out_dir"], "thresholding"); os.makedirs(out_dir, exist_ok=True)
    thr_summary = []
    for name, r in results.items():
        y_true = r["test"]["y_true"].astype(int)
        y_score = r["test"]["y_pred"].astype(float)
        thresholds = np.linspace(0, 1, 501)
        precs, recs, f1s, tprs, fprs = [], [], [], [], []
        for t in thresholds:
            y_hat = (y_score >= t).astype(int)
            tp = ((y_hat==1)&(y_true==1)).sum()
            fp = ((y_hat==1)&(y_true==0)).sum()
            tn = ((y_hat==0)&(y_true==0)).sum()
            fn = ((y_hat==0)&(y_true==1)).sum()
            prec = tp/(tp+fp+1e-9); rec  = tp/(tp+fn+1e-9)
            f1   = 2*prec*rec/(prec+rec+1e-9)
            tpr  = rec; fpr  = fp/(fp+tn+1e-9)
            precs.append(prec); recs.append(rec); f1s.append(f1); tprs.append(tpr); fprs.append(fpr)
        precs = np.array(precs); recs = np.array(recs); f1s = np.array(f1s); tprs = np.array(tprs); fprs = np.array(fprs)
        t_opt_f1 = thresholds[np.nanargmax(f1s)]
        youden = tprs - fprs; t_opt_j = thresholds[np.nanargmax(youden)]
        thr_summary.append({"model": name, "t_f1": float(t_opt_f1), "t_youden": float(t_opt_j)})
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.plot(thresholds, precs, label="Precision"); plt.plot(thresholds, recs, label="Recall")
        plt.title(f"{name}: Precision/Recall vs Threshold"); plt.xlabel("Threshold"); plt.grid(True, alpha=0.3); plt.legend()
        plt.subplot(1,3,2); plt.plot(thresholds, f1s, color="#ff7f0e"); plt.axvline(t_opt_f1, color="k", ls="--", label=f"t* F1={t_opt_f1:.3f}")
        plt.title(f"{name}: F1 vs Threshold"); plt.xlabel("Threshold"); plt.legend(); plt.grid(True, alpha=0.3)
        fnrs = 1 - tprs; plt.subplot(1,3,3); plt.plot(fprs, fnrs)
        plt.title(f"{name}: DET (FNR vs FPR)"); plt.xlabel("FPR"); plt.ylabel("FNR"); plt.grid(True, alpha=0.3)
        out_p = os.path.join(out_dir, f"{name}_thresholding.png"); plt.tight_layout(); plt.savefig(out_p, dpi=150); plt.close()
        print("Saved:", out_p)
    thr_df = pd.DataFrame(thr_summary); thr_csv = os.path.join(out_dir, "thresholds.csv"); thr_df.to_csv(thr_csv, index=False)
    print("Thresholds CSV:", thr_csv)
