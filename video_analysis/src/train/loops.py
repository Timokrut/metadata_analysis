import numpy as np, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def get_optimizer(model, cfg):
    T = cfg["train"]
    if T["optimizer"].lower()=="adamw":
        return torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    return torch.optim.SGD(model.parameters(), lr=T["lr"], momentum=0.9, nesterov=True, weight_decay=T["weight_decay"])

def get_scheduler(optimizer, cfg):
    T = cfg["train"]
    sch = T.get("scheduler")
    if sch == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T["epochs"])
    if sch == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    return None

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, amp=True):
    model.eval()
    loss_meter, y_true, y_pred = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_meter.append(loss.item())
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float('nan')
    ap = average_precision_score(y_true, y_pred)
    preds_bin = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_bin, labels=[0,1])
    acc = (preds_bin == y_true).mean()
    return {"loss": float(np.mean(loss_meter)),"auc": float(auc),"ap": float(ap),"acc": float(acc),
            "cm": cm,"y_true": y_true,"y_pred": y_pred}

def train_epoch(model, loader, criterion, optimizer, device, amp=True):
    model.train()
    loss_meter, y_true, y_pred = [], [], []
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()
        loss_meter.append(loss.item())
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float('nan')
    ap = average_precision_score(y_true, y_pred)
    acc = ((y_pred >= 0.5).astype(int) == y_true).mean()
    return {"loss": float(np.mean(loss_meter)),"auc": float(auc),"ap": float(ap),"acc": float(acc)}

def train_model(model_name, model_ctor, cfg, train_loader, val_loader, pos_weight=None, device="cuda"):
    T = cfg["train"]; P = cfg["paths"]
    model = model_ctor().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    history, best_auc, best_path = [], -1.0, f"{P['out_dir']}/{model_name}_best.pt"
    patience = T["early_stop_patience"]; epochs_no_improve = 0

    for epoch in range(1, T["epochs"]+1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device, amp=T["amp"])
        vl = evaluate_epoch(model, val_loader, criterion, device, amp=T["amp"])
        if scheduler is not None:
            if T["scheduler"] == "plateau": scheduler.step(vl["auc"])
            else: scheduler.step()
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_auc": tr["auc"],
               "val_loss": vl["loss"], "val_auc": vl["auc"], "val_acc": vl["acc"], "val_ap": vl["ap"]}
        history.append(row)
        print(f"[{model_name}] Epoch {epoch:02d}/{T['epochs']}  train_loss={tr['loss']:.4f} val_loss={vl['loss']:.4f}  train_auc={tr['auc']:.4f} val_auc={vl['auc']:.4f}  val_acc={vl['acc']:.4f}")
        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            os.makedirs(P["out_dir"], exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    import pandas as pd
    hist_df = pd.DataFrame(history)
    return model, best_path, hist_df
