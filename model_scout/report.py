import os

def make_html_report(outdir, plots, meta, ranked_df, df_results):
    report_dir = os.path.join(outdir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "summary.html")

    def rel(p): return os.path.relpath(p, report_dir).replace("\\", "/")

    top_html = ranked_df.head(20).to_html(index=False, float_format=lambda x: f"{x:.4f}")

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
<p><b>Task:</b> {meta['task']} | <b>α:</b> {meta['alpha']} | <b>Jobs:</b> {meta['n_jobs']}</p>
<p><b>Models:</b> {', '.join(meta['models'])}<br>
<b>Encodings:</b> {', '.join(meta['encodings'])}<br>
<b>Sample grid:</b> {', '.join(map(str, meta['sample_grid']))}<br>
<b>Total runs:</b> {len(df_results)}</p>

<h2>Top Configurations (Spearman ρ)</h2>
{top_html}

<h2>Plots</h2>
<div class="plots">
  <div class="card"><b>Heatmap</b><a href="{rel(plots['heatmap'])}" target="_blank"><img class="thumb" src="{rel(plots['heatmap'])}"/></a></div>
  <div class="card"><b>ρ vs Samples</b><a href="{rel(plots['rho_vs_samples'])}" target="_blank"><img class="thumb" src="{rel(plots['rho_vs_samples'])}"/></a></div>
  <div class="card"><b>Runtime per Model</b><a href="{rel(plots['runtime'])}" target="_blank"><img class="thumb" src="{rel(plots['runtime'])}"/></a></div>
</div>

<p class="small">Report generated in {outdir}</p>
</body></html>"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path
