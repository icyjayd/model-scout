import os
from datetime import datetime


def make_html_report(outdir, plots, meta, ranked_df, df_results):
    """
    Generate an HTML report summarizing Model Scout results.
    Now dynamically labels ranking metric (accuracy or Spearman ρ)
    and includes generation timestamp.
    """

    report_dir = os.path.join(outdir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "summary.html")

    def rel(p):  # relative path helper for embedded links
        return os.path.relpath(p, report_dir).replace("\\", "/")

    # Detect ranking metric type from ranked_df
    if "metric_type" in ranked_df.columns and not ranked_df.empty:
        metric_label = (
            "accuracy"
            if ranked_df["metric_type"].iloc[0] == "accuracy"
            else "Spearman ρ"
        )
    else:
        # Fallback for backward compatibility
        metric_label = "Spearman ρ"

    # Top table of ranked results
    top_html = ranked_df.head(20).to_html(index=False, float_format=lambda x: f"{x:.4f}")

    # Format metadata
    models = ", ".join(meta.get("models", [])) if isinstance(meta.get("models"), list) else meta.get("models", "")
    encodings = ", ".join(meta.get("encodings", [])) if isinstance(meta.get("encodings"), list) else meta.get("encodings", "")
    sample_grid = ", ".join(map(str, meta.get("sample_grid", []))) if isinstance(meta.get("sample_grid"), (list, tuple)) else meta.get("sample_grid", "")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8" />
<title>Model Scout Report</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;}}
h1{{margin-bottom:8px;}} h2{{margin-top:28px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:0.95rem;}}
th{{background:#f7f7f7;}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:12px;}}
.card{{border:1px solid #eee;border-radius:10px;padding:10px;background:#fff;}}
.thumb{{width:100%;height:auto;border-radius:6px;border:1px solid #eee;}}
.small{{color:#666;font-size:0.9rem;}}
</style></head><body>
<h1>Model Scout Report</h1>
<p><b>Task:</b> {meta.get('task','?')} | <b>α:</b> {meta.get('alpha','?')} | <b>Jobs:</b> {meta.get('n_jobs','?')}</p>
<p><b>Models:</b> {models}<br>
<b>Encodings:</b> {encodings}<br>
<b>Sample grid:</b> {sample_grid}<br>
<b>Total runs:</b> {len(df_results)}</p>

<h2>Top Configurations (ranked by {metric_label})</h2>
{top_html}

<h2>Plots</h2>
<div class="plots">
  <div class="card"><b>Heatmap</b><a href="{rel(plots['heatmap'])}" target="_blank"><img class="thumb" src="{rel(plots['heatmap'])}"/></a></div>
  <div class="card"><b>ρ vs Samples</b><a href="{rel(plots['rho_vs_samples'])}" target="_blank"><img class="thumb" src="{rel(plots['rho_vs_samples'])}"/></a></div>
  <div class="card"><b>Runtime per Model</b><a href="{rel(plots['runtime'])}" target="_blank"><img class="thumb" src="{rel(plots['runtime'])}"/></a></div>
</div>

<p class="small">Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} in {outdir}</p>
</body></html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
