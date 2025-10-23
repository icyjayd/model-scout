"""
model_scout.main

Entry point and orchestrator for Model Scout.
Now includes:
 - Progressive CSV saving after each model instance
 - Automatic saving of best-performing models to /models folder
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import joblib
from joblib import Parallel, delayed

from . import (
    config,
    data_utils,
    run_single,
    models,
    encoding,
    aggregator,
    plotting,
    report,
)


def run_scout(
    sequences,
    labels,
    models_list,
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
    out_dir = out_json.parent
    out_csv = out_json.with_suffix(".csv")
    out_csv.unlink(missing_ok=True)

    print(f"📁 Results will be progressively written to: {out_csv.resolve()}")

    total_jobs = len(models_list) * len(encodings) * len(sample_sizes)
    results = []

    def _save_progress(result):
        """Append a single result dict to CSV progressively."""
        df_row = pd.DataFrame([result])
        write_header = not out_csv.exists() or out_csv.stat().st_size == 0
        df_row.to_csv(out_csv, mode="a", header=write_header, index=False)

    print(f"🚀 Running {total_jobs} model instances ({n_jobs} parallel jobs)...")

    combos = [
        (m, e, n)
        for m in models_list
        for e in encodings
        for n in sample_sizes
    ]

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

    for i, result in enumerate(parallel_results, 1):
        results.append(result)
        _save_progress(result)
        print(
            f"[{i}/{total_jobs}] ✅ {result.get('model')} | {result.get('encoding')} | n={result.get('n_samples')} saved."
        )

    # Aggregate and save final results
    print("💾 Aggregating and saving final results...")
    df_results = aggregator.save_results(results, out_json)

    ranked = aggregator.rank_results(df_results, alpha)
    print("\n🏆 Top-ranked combinations:")
    print(ranked.head(10).to_string(index=False))

    # --- 🔽 Save best-performing models per type ---
    print("\n🧠 Training and saving best-performing models...")
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for model_name in df_results["model"].unique():
        df_model = df_results[df_results["model"] == model_name]
        if df_model.empty:
            continue
        # pick best based on Spearman rho
        best_row = df_model.sort_values("rho", ascending=False).iloc[0]
        best_encoding = best_row["encoding"]
        best_n = int(best_row["n_samples"])

        print(f"  → {model_name}: best with {best_encoding} (n={best_n})")

        # Retrain on all available data
        seqs = df["sequence"].tolist()[:best_n]
        labels_subset = df["label"].tolist()[:best_n]
        X = encoding.encode_sequences(seqs, best_encoding)
        y = labels_subset

        model = models.build_model(task, model_name, seed)
        model.fit(X, y)

        # Save model
        model_path = models_dir / f"{model_name}_{best_encoding}_{best_n}.joblib"
        joblib.dump(model, model_path)
        print(f"     💾 Saved to {model_path.name}")

    # --- 🔼 Done saving models ---

    # Plots
    print("\n📊 Generating plots...")
    plot_paths = plotting.make_plots(df_results, out_dir)

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
    report.make_html_report(out_dir, plot_paths, meta, ranked, df_results)

    summary = {
        "out_json": str(out_json.resolve()),
        "out_csv": str(out_csv.resolve()),
        "models_dir": str(models_dir.resolve()),
        "plots": plot_paths,
        "meta": meta,
        "top": ranked.head(20).to_dict(orient="records"),
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All done!")
    print(f"Summary JSON: {out_json.resolve()}")
    print(f"Progressive CSV: {out_csv.resolve()}")
    print(f"Saved models: {models_dir.resolve()}")
    print(f"HTML Report: {out_dir / 'reports' / 'summary.html'}")

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
        models_list=args.models,
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
