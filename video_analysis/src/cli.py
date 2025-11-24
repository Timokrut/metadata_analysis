import typer, yaml
from pathlib import Path

app = typer.Typer(help="MesoNet FF++")

def load_cfg(config: Path):
    with open(config, "r") as f:
        return yaml.safe_load(f)

@app.command("ffpp-download")
def ffpp_download(kaggle_id: str = typer.Option("xdxd003/ff-c23"),
                  dst: Path = typer.Option(Path("data/ffpp"), file_okay=False)):
    from src.data.ffpp import download_ffpp_kagglehub
    dst.mkdir(parents=True, exist_ok=True)
    root = download_ffpp_kagglehub(kaggle_id, dst)
    typer.echo(f"FF++ downloaded to: {root}")

@app.command("ffpp-discover")
def ffpp_discover(config: Path = typer.Option(Path("configs/train.yaml"), exists=True),
                  out: Path = typer.Option(Path("data/ffpp_manifest.csv"))):
    from src.data.ffpp import discover_ffpp, make_splits
    cfg = load_cfg(config)
    df = discover_ffpp(cfg)
    df = make_splits(cfg, df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"Manifest saved: {out} | total={len(df)}")

@app.command("ffpp-extract-faces")
def ffpp_extract_faces(config: Path = typer.Option(Path("configs/train.yaml"), exists=True),
                       manifest: Path = typer.Option(Path("data/ffpp_manifest.csv"), exists=True),
                       device: str = typer.Option("cuda")):
    import pandas as pd
    from src.data.ffpp import extract_ffpp_crops
    cfg = load_cfg(config)
    df = pd.read_csv(manifest)
    extract_ffpp_crops(df, cfg, device=device)
    typer.echo(f"Faces saved to: {cfg['dataset']['crops_cache_dir']}")

@app.command("train-both")
def train_both(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import train_both_models
    cfg = load_cfg(config)
    train_both_models(cfg)

@app.command("build-report")
def build_report(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import load_best_results
    from src.viz.plots import build_infographic
    from src.reports.report import build_html_report, save_metrics_csv
    cfg = load_cfg(config)
    results, histories, infographic_path, metrics_path = load_best_results(cfg)
    build_infographic(histories, results, infographic_path, cfg)
    html_path = build_html_report(results, infographic_path, metrics_path, cfg)
    typer.echo(f"Report saved: {html_path}")

@app.command("interpret-occlusion")
def interpret_occlusion(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import load_best_models_and_test
    from src.analysis.interpret import run_interpretability
    cfg = load_cfg(config)
    models, test_paths, test_labels = load_best_models_and_test(cfg)
    run_interpretability(models, test_paths, test_labels, cfg)

@app.command("threshold-curves")
def threshold_curves_cmd(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import load_results_only
    from src.analysis.thresholds import run_threshold_curves
    cfg = load_cfg(config)
    results = load_results_only(cfg)
    run_threshold_curves(results, cfg)

@app.command("error-grids")
def error_grids_cmd(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import load_best_models_and_test
    from src.analysis.errors import run_error_grids
    cfg = load_cfg(config)
    models, test_paths, test_labels = load_best_models_and_test(cfg)
    run_error_grids(models, test_paths, test_labels, cfg)

@app.command("export-models")
def export_models_cmd(config: Path = typer.Option(Path("configs/train.yaml"), exists=True)):
    from src.train.runner import load_best_models
    from src.export.export import export_models
    cfg = load_cfg(config)
    models = load_best_models(cfg)
    export_models(models, cfg)

if __name__ == "__main__":
    app()
