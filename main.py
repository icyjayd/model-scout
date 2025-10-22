# --- Auto-detect your repo root so we can import ml_models.* ---
# import sys
# from pathlib import Path
# if "ml_models" not in sys.modules:
#     here = Path(__file__).resolve()
#     for parent in here.parents:
#         if (parent / "ml_models").exists():
#             sys.path.insert(0, str(parent))
#             break
# ---------------------------------------------------------------

import os, json
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_MODELS, DEFAULT_ENCODINGS, DEFAULT_SAMPLE_GRID, ALPHA, N_JOBS
from .run_single import run_single
from .aggregator import save_results, rank_results
from .plotting import make_plots
from .report import make_html_report
from .data_utils import load_data  # from your main project

def run_scout(
    seq_file,
    labels_file=None,
    models=DEFAULT_MODELS,
    encodings=DEFAULT_ENCODINGS,
    sample_grid=DEFAULT_SAMPLE_GRID,
    alpha=ALPHA,
    seed=42,
    test_size=0.2,
    stratify="none",
    outpath="runs/model_scout/model_scout_results.json",
    n_jobs=N_JOBS,
):
    print(f"[INFO] Starting model scout using {n_jobs} parallel jobs")

    df = load_data(seq_file, labels_file)
    
    if not df["label"].dtype.kind in "if":
        df["label"] = LabelEncoder().fit_transform(df["label"])
        task = "classification"
    else:
        task = "regression"
    # --- Sanity check: preview first 10 sequences and labels ---
    try:
        n_preview = min(10, len(df))
        seq_preview = [str(s)[:10] for s in df["sequence"].head(n_preview)]
        label_preview = df["label"].head(n_preview).tolist()
        print("[INFO] Data preview (first 10):")
        for i, (seq, label) in enumerate(zip(seq_preview, label_preview)):
            print(f"  {i+1:2d}. {seq:<12} → {label}")
    except Exception as e:
        print(f"[WARN] Could not preview sequences/labels: {e}")

    combos = [
        (m, e, n)
        for m in models for e in encodings for n in sample_grid
        if len(df) >= n
    ]
    print(f"[INFO] Total runs: {len(combos)}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_single)(m, e, n, df, task, seed, test_size, stratify)
        for (m, e, n) in combos
    )

    outdir = Path(outpath).parent
    df_results = save_results(results, outpath)
    grouped = rank_results(df_results, alpha)
    print("\nTop configurations (Spearman ρ):")
    print(grouped.head(10).to_string(index=False))

    plots = make_plots(df_results, os.path.join(outdir, "plots"))
    report_path = make_html_report(
        outdir=str(outdir),
        plots=plots,
        meta={
            "alpha": alpha,
            "models": models,
            "encodings": encodings,
            "sample_grid": sample_grid,
            "n_jobs": n_jobs,
            "task": task,
        },
        ranked_df=grouped,
        df_results=df_results,
    )

    final = {
        "alpha": alpha,
        "models_tested": models,
        "encodings_tested": encodings,
        "sample_grid": sample_grid,
        "n_jobs": n_jobs,
        "task": task,
        "ranked_results": grouped.to_dict(orient="records"),
        "all_results": results,
        "plot_dir": os.path.join(outdir, "plots"),
        "html_report": report_path,
    }
    with open(outpath, "w") as f:
        json.dump(final, f, indent=2)

    print(f"[INFO] Report: {report_path}")
    print(f"[INFO] Results: {outpath}")
    return final

def cli_entry():
    import argparse
    ap = argparse.ArgumentParser(description="Parallel model scout (modular).")
    ap.add_argument("sequences", type=str)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--encodings", type=str, default=",".join(DEFAULT_ENCODINGS))
    ap.add_argument("--samples", type=str, default=",".join(map(str, DEFAULT_SAMPLE_GRID)))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--stratify", type=str, default="none", choices=["none", "auto"])
    ap.add_argument("--out", type=str, default="runs/model_scout/model_scout_results.json")
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    sample_grid = [int(x) for x in args.samples.split(",") if x.strip()]

    run_scout(
        seq_file=args.sequences,
        labels_file=args.labels,
        models=models,
        encodings=encodings,
        sample_grid=sample_grid,
        alpha=args.alpha,
        seed=args.seed,
        test_size=args.test_size,
        stratify=args.stratify,
        outpath=args.out,
        n_jobs=args.jobs,
    )
