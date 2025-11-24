import os, torch

def export_models(models: dict, cfg: dict):
    out_dir = os.path.join(cfg["paths"]["out_dir"], "export"); os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1,3,cfg["input"]["image_size"],cfg["input"]["image_size"], device=next(iter(models.values())).parameters().__next__().device)
    summary = []
    for name, m in models.items():
        m.eval()
        # TorchScript
        ts_path = os.path.join(out_dir, f"{name}.ts")
        traced = torch.jit.trace(m, dummy); traced.save(ts_path)
        # ONNX
        onnx_path = os.path.join(out_dir, f"{name}.onnx")
        torch.onnx.export(m, dummy, onnx_path, input_names=["input"], output_names=["logits"],
                          opset_version=13, do_constant_folding=True,
                          dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
        summary.append({"model": name, "torchscript": ts_path, "onnx": onnx_path})
        print(f"Exported {name} → {ts_path} | {onnx_path}")
    # можно добавить запись в session.json при желании
    return summary
