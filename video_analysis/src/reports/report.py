import os, time, pandas as pd

def save_metrics_csv(metrics_df, cfg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(cfg["paths"]["out_dir"], f"metrics_{ts}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_path

def build_html_report(results, infographic_path, metrics_path, cfg):
    out_dir = cfg["paths"]["out_dir"]; html_path = os.path.join(out_dir, f"report_{os.path.basename(infographic_path).split('_')[-1].split('.')[0]}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>MesoNet Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial; margin: 20px; }}
h1,h2,h3 {{ margin-top: 1.2em; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; }}
img {{ max-width: 100%; height: auto; border:1px solid #eee; }}
.code {{ background:#f6f8fa; padding:8px; border-radius:6px; font-family:ui-monospace,Consolas,Monaco,monospace; }}
</style></head><body>
<h1>Отчёт об обучении Meso4 и MesoInception4</h1>
<p>Дата/время: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
<h2>Итоговые метрики (test)</h2>
{pd.read_csv(metrics_path).to_html(index=False)}
<h2>Инфографика</h2>
<img src="{os.path.basename(infographic_path)}" alt="infographic"/>
<h2>Отдельные графики</h2>
<ul>
<li>Learning curves: <span class="code">figs/learning_curves.png</span></li>
<li>ROC&PR: <span class="code">figs/curves_roc_pr.png</span></li>
<li>Confusion matrices: <span class="code">figs/confusion_cm.png</span></li>
<li>Calibration: <span class="code">figs/calib_calibration.png</span></li>
<li>Score histograms: <span class="code">figs/hist_score_hist.png</span></li>
</ul>
</body></html>
""")
    return html_path
