"""
model_scout.main

Entry point and orchestrator for Model Scout.
Now includes progressive CSV saving after each model instance run.
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

from . import (
    config,
    data_utils,
    run_single,
    aggregator,
    plotting,
    report,
)


def run_scout(
    sequences,
    labels,
    models,
    encodings,
    sample_sizes,
    task="regression",
    alpha=config.ALPHA,
    seed=42,
    test_size=0.2,
    n_jobs=config.N_JOBS,
    outpath="model_scout_results.json",
):
    """Main driver for Model Scout pipeline."""

    print("🔍 Loading data...")
    df = data_utils.load_data(sequences, labels)
    print(f"Loaded {len(df)} sequences with labels.")

    # Output paths
    out_json = Path(outpath)
    out_csv = out_json.with_suffix(".csv")
    out_csv.unlink(missing_ok=True)

    print(f"📁 Results will be progressively written to: {out_csv.resolve()}")

    total_jobs = len(models) * len(encodings) * len(sample_sizes)
    results = []

    def _save_progress(result):
        """Append a single result dict to CSV progressively."""
        df_row = pd.DataFrame([result])
        # header only if file doesn't exist or empty
        write_header = not out_csv.exists() or out_csv.stat().st_size == 0
        df_row.to_csv(out_csv, mode="a", header=write_header, index=False)

    print(f"🚀 Running {total_jobs} model instances ({n_jobs} parallel jobs)...")

    combos = [
        (m, e, n)
        for m in models
        for e in encodings
        for n in sample_sizes
    ]

    # Run all combinations in parallel
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(run_single.run_single)(
            df["sequence"].tolist(),
            df["label"].tolist(),
            model_name=m,
            encoding=e,
            n_samples=n,
            task=task,
            test_size=test_size,
            seed=seed,
        )
        for (m, e, n) in combos
    )

    # Save progressively as they finish
    for i, result in enumerate(parallel_results, 1):
        results.append(result)
        _save_progress(result)
        print(
            f"[{i}/{total_jobs}] ✅ {result.get('model')} | {result.get('encoding')} | n={result.get('n_samples')} saved."
        )

    # Save full results
    print("💾 Aggregating and saving final results...")
    df_results = aggregator.save_results(results, out_json)

    # Ranking
    ranked = aggregator.rank_results(df_results, alpha)
    print("\n🏆 Top-ranked combinations:")
    print(ranked.head(10).to_string(index=False))

    # Plots
    print("\n📊 Generating plots...")
    plot_paths = plotting.make_plots(df_results, out_json.parent)

    # Report
    print("\n🧾 Creating HTML report...")
    meta = {
        "task": task,
        "alpha": alpha,
        "seed": seed,
        "test_size": test_size,
        "n_jobs": n_jobs,
        "total_runs": total_jobs,
    }
    report.make_html_report(out_json.parent, plot_paths, meta, ranked, df_results)

    summary = {
        "out_json": str(out_json.resolve()),
        "out_csv": str(out_csv.resolve()),
        "plots": plot_paths,
        "meta": meta,
        "top": ranked.head(20).to_dict(orient="records"),
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All done!")
    print(f"Summary JSON: {out_json.resolve()}")
    print(f"Progressive CSV: {out_csv.resolve()}")
    print(f"HTML Report: {out_json.parent / 'reports' / 'summary.html'}")

    return summary


def cli_entry():
    """Command-line interface for Model Scout."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple models and encodings on sequence-property data."
    )
    parser.add_argument("sequences", help="Path to sequences .csv or .npy file")
    parser.add_argument("--labels", required=True, help="Path to labels .csv or .npy file")
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        help=f"Models to test (default: {config.DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=config.DEFAULT_ENCODINGS,
        help=f"Encodings to test (default: {config.DEFAULT_ENCODINGS})",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=config.DEFAULT_SAMPLE_GRID,
        help=f"Sample sizes (default: {config.DEFAULT_SAMPLE_GRID})",
    )
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--out", default="model_scout_results.json")

    args = parser.parse_args()

    run_scout(
        sequences=args.sequences,
        labels=args.labels,
        models=args.models,
        encodings=args.encodings,
        sample_sizes=args.samples,
        task=args.task,
        alpha=args.alpha,
        seed=args.seed,
        test_size=args.test_size,
        n_jobs=args.jobs,
        outpath=args.out,
    )


if __name__ == "__main__":
    cli_entry()
