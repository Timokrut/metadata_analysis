import os, cv2, numpy as np
from math import ceil, sqrt

def _save_grid(image_paths, out_path, n_max=36, cell=256, pad=4, color=(0,0,255)):
    sel = image_paths[:n_max]
    if len(sel)==0: return False
    n = len(sel); cols = int(ceil(sqrt(n))); rows = int(ceil(n/cols))
    H = rows*cell + (rows+1)*pad; W = cols*cell + (cols+1)*pad
    canvas = np.ones((H,W,3), dtype=np.uint8)*245
    for i, p in enumerate(sel):
        r = i//cols; c = i%cols
        bgr = cv2.imread(p); if bgr is None: continue
        bgr = cv2.resize(bgr, (cell, cell), interpolation=cv2.INTER_AREA)
        y0 = r*cell + (r+1)*pad; x0 = c*cell + (c+1)*pad
        canvas[y0:y0+cell, x0:x0+cell] = bgr
        cv2.rectangle(canvas, (x0, y0), (x0+cell-1, y0+cell-1), color, 2)
    cv2.imwrite(out_path, canvas); return True

def run_error_grids(models: dict, test_paths, test_labels, cfg):
    out_dir = os.path.join(cfg["paths"]["out_dir"], "errors"); os.makedirs(out_dir, exist_ok=True)
    # здесь проще перечитать preds из results при необходимости; для краткости строим по 0.5
    from src.train.runner import load_results_only
    results = load_results_only(cfg)
    for name, r in results.items():
        y_true = r["test"]["y_true"].astype(int); y_score = r["test"]["y_pred"].astype(float)
        y_hat = (y_score >= 0.5).astype(int)
        fp_idx = [i for i in range(len(y_true)) if y_hat[i]==1 and y_true[i]==0]
        fn_idx = [i for i in range(len(y_true)) if y_hat[i]==0 and y_true[i]==1]
        fp_paths = [test_paths[i] for i in fp_idx]; fn_paths = [test_paths[i] for i in fn_idx]
        fp_grid = os.path.join(out_dir, f"{name}_FP_grid.png"); fn_grid = os.path.join(out_dir, f"{name}_FN_grid.png")
        ok_fp = _save_grid(fp_paths, fp_grid, color=(0,0,255)); ok_fn = _save_grid(fn_paths, fn_grid, color=(255,0,0))
        print(f"{name}: FP={len(fp_paths)} FN={len(fn_paths)}"); 
        if ok_fp: print("Saved:", fp_grid)
        if ok_fn: print("Saved:", fn_grid)
