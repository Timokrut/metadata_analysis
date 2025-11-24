import os, numpy as np, cv2, torch
import matplotlib.cm as cm
from src.data.datasets import AlbumentationsTransform

def occlusion_map(model, img_tensor, device, patch=16, stride=8):
    model.eval()
    with torch.no_grad():
        base = torch.sigmoid(model(img_tensor.unsqueeze(0).to(device))).item()
    c, H, W = img_tensor.shape
    grid_h = (H - patch)//stride + 1; grid_w = (W - patch)//stride + 1
    scores = np.zeros((grid_h, grid_w), dtype=np.float32)
    with torch.no_grad():
        for i, y in enumerate(range(0, H - patch + 1, stride)):
            for j, x in enumerate(range(0, W - patch + 1, stride)):
                img_mod = img_tensor.clone()
                img_mod[:, y:y+patch, x:x+patch] = 0.0
                s = torch.sigmoid(model(img_mod.unsqueeze(0).to(device))).item()
                scores[i, j] = base - s
    return scores, base

def overlay_heatmap(rgb, heat, alpha=0.45, cmap_name='jet'):
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
    heat_color = cm.get_cmap(cmap_name)(heat)[..., :3]
    overlay = (1 - alpha) * (rgb/255.0) + alpha * heat_color
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay

def run_interpretability(models: dict, test_paths, test_labels, cfg):
    out_dir = os.path.join(cfg["paths"]["out_dir"], "interpretability"); os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_tf = AlbumentationsTransform(augment=False, size=cfg["input"]["image_size"])
    # выберем по одному примеру на класс
    real_idx = [i for i,y in enumerate(test_labels) if y==0]
    fake_idx = [i for i,y in enumerate(test_labels) if y==1]
    sel = []
    import numpy as np
    rng = np.random.RandomState(cfg["split"]["random_seed"]+123)
    if len(real_idx)>0: sel.extend(rng.choice(real_idx, size=min(1, len(real_idx)), replace=False).tolist())
    if len(fake_idx)>0: sel.extend(rng.choice(fake_idx, size=min(1, len(fake_idx)), replace=False).tolist())

    for name, model in models.items():
        for idx in sel:
            p = test_paths[idx]; y = test_labels[idx]
            bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = eval_tf(rgb); img_t = torch.from_numpy(img).permute(2,0,1).float()
            occ, base = occlusion_map(model, img_t, device, patch=16, stride=8)
            H, W = img_t.shape[1], img_t.shape[2]
            occ_up = cv2.resize(occ, (W, H), interpolation=cv2.INTER_CUBIC)
            overlay = overlay_heatmap(rgb, occ_up, alpha=0.5, cmap_name='jet')
            canvas = np.hstack([rgb, overlay])
            out_path = os.path.join(out_dir, f"{name}_occ_{os.path.basename(p)}")
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print("Saved:", out_path)
