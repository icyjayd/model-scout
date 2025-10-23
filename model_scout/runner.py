"""
runner.py
Orchestrates the Model Scout benchmarking pipeline.
Usable as a Python function (`run_scout`) or from the CLI.
"""

import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import joblib
from datetime import datetime
from . import config, data_utils, run_single, models, encoding, aggregator, plotting, report
from .utils import progress


# --- internal helpers -------------------------------------------------------



def _run_parallel_jobs(df, models_list, encodings, sample_sizes, task, seed, test_size, stratify, n_jobs, out_csv, model_config_path=None):
    """Run all (model, encoding, n_samples) combinations in parallel with clean, pickle-safe logging."""
    combos = [(m, e, n) for m in models_list for e in encodings for n in sample_sizes]
    total_jobs = len(combos)
    print(f"🚀 Starting {total_jobs} experiments ({n_jobs} parallel jobs)")

    def run_and_log(job_index, model_name, encoding, n_samples):
        """Worker job: run a single experiment."""
        print(f"\n▶ ({job_index + 1}/{total_jobs}) Running model={model_name} | encoding={encoding} | "
              f"n_samples={n_samples} | task={task} | seed={seed} | stratify={stratify}")
        start_time = datetime.now()
        try:
            result = run_single.run_single(
                model_name=model_name,
                encoding=encoding,
                n_samples=n_samples,
                df=df,
                task=task,
                seed=seed,
                test_size=test_size,
                stratify=stratify,
                model_config_path=model_config_path,
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"✅ ({job_index + 1}/{total_jobs}) Completed model={model_name} | encoding={encoding} | "
                  f"n_samples={n_samples} | ρ={result.get('rho', float('nan')):.4f} | "
                  f"p={result.get('p', 'N/A')} | time={elapsed:.2f}s | status={result.get('status', 'ok')}")
            return result
        except Exception as e:
            print(f"❌ ({job_index + 1}/{total_jobs}) Failed model={model_name} | encoding={encoding} | "
                  f"n_samples={n_samples} | error={e}")
            return {
                "model": model_name,
                "encoding": encoding,
                "n_samples": n_samples,
                "rho": float("nan"),
                "p": float("nan"),
                "seconds": float("nan"),
                "status": f"error: {e}",
            }

    # Run jobs in parallel
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(run_and_log)(i, m, e, n) for i, (m, e, n) in enumerate(combos)
    )

    # Save progressively and track finished count
    for i, result in enumerate(parallel_results, 1):
        progress.append_to_csv(result, out_csv)
        print(f"[{i}/{total_jobs}] Saved {result['model']} | {result['encoding']} | n={result['n_samples']}")

    print(f"\n🏁 All {total_jobs} experiments finished successfully.")
    return parallel_results

def _train_best_models(df_results, df, task, seed, out_dir, model_config_path=None):
    """Retrain and save the best performer for each model type."""
    print("\n🧠 Retraining best-performing models...")
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for model_name in df_results["model"].unique():
        df_model = df_results[df_results["model"] == model_name]
        if df_model.empty:
            continue

        metric = "accuracy" if "accuracy" in df_model.columns and df_model["accuracy"].notna().any() else "rho"
        best_row = df_model.sort_values(metric, ascending=False).iloc[0]

        best_encoding = best_row["encoding"]
        best_n = int(best_row["n_samples"])
        print(f"  → {model_name}: best with {best_encoding} (n={best_n})")

        seqs = df["sequence"].tolist()[:best_n]
        labels_subset = df["label"].tolist()[:best_n]
        X = encoding.encode_sequences(seqs, best_encoding)
        y = labels_subset

        model = models.build_model(task, model_name, seed, model_config_path=model_config_path)
        model.fit(X, y)
        model_path = models_dir / f"{model_name}_{best_encoding}_{best_n}.joblib"
        joblib.dump(model, model_path)
        print(f"     💾 Saved {model_path.name}")

    return models_dir


def _generate_report(out_dir, df_results, ranked, task, alpha, seed, test_size, stratify, n_jobs, models_list, encodings, sample_sizes):
    """Generate plots and an HTML report."""
    print("\n📊 Generating plots and HTML report...")
    plot_paths = plotting.make_plots(df_results, out_dir)
    meta = {
        "task": task,
        "alpha": alpha,
        "seed": seed,
        "test_size": test_size,
        "stratify": stratify,
        "n_jobs": n_jobs,
        "models": models_list,
        "encodings": encodings,
        "sample_grid": sample_sizes,
        "total_runs": len(df_results),
    }
    report.make_html_report(out_dir, plot_paths, meta, ranked, df_results)
    return plot_paths, meta


# --- main function ----------------------------------------------------------

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
    stratify=False,
    n_jobs=config.N_JOBS,
    outpath="model_scout_results/",
    model_config_path=None,
):
    """Run the full Model Scout pipeline."""

    print("🔍 Loading data...")
    df = data_utils.load_data(sequences, labels)
    print(f"Loaded {len(df)} sequences with labels.")

    # --- Handle --out as directory always ---
    out_dir = Path(outpath)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "results.json"
    out_csv = out_dir / "results.csv"

    print(f"📁 Output directory: {out_dir.resolve()}")
    print(f"📊 Progressive results → {out_csv.resolve()}")

    # Clean old results if re-running
    out_csv.unlink(missing_ok=True)

    # --- main experiment loop ---
    results = _run_parallel_jobs(
        df,
        models_list,
        encodings,
        sample_sizes,
        task,
        seed,
        test_size,
        stratify,
        n_jobs,
        out_csv,
        model_config_path=model_config_path,
    )

    print("💾 Aggregating final results...")
    df_results = aggregator.save_results(results, out_json)
    ranked = aggregator.rank_results(df_results, alpha)

    print("\n🏆 Top-ranked combinations:")
    print(ranked.head(10).to_string(index=False))

    # Save best models
    models_dir = _train_best_models(df_results, df, task, seed, out_dir, model_config_path)

    # Generate plots and report
    plot_paths, meta = _generate_report(
        out_dir,
        df_results,
        ranked,
        task,
        alpha,
        seed,
        test_size,
        stratify,
        n_jobs,
        models_list,
        encodings,
        sample_sizes,
    )

    summary = {
        "out_json": str(out_json.resolve()),
        "out_csv": str(out_csv.resolve()),
        "models_dir": str(models_dir.resolve()),
        "plots": plot_paths,
        "meta": meta,
        "top": ranked.head(20).to_dict(orient="records"),
    }

    progress.write_summary(summary, out_json)
    print(f"\n✅ All done! Report and outputs saved in {out_dir.resolve()}")
    return summary
