"""
runner.py
Orchestrates the Model Scout benchmarking pipeline.
Usable as a Python function (`run_scout`) or from the CLI.
"""

import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import joblib
import time
from datetime import datetime
from . import config, data_utils, run_single, aggregator, plotting, report
from .utils import progress
from .models import build_model
from .encoding import encode_sequences
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --- internal helpers -------------------------------------------------------



def _run_parallel_jobs(
    df, models_list, encodings, sample_sizes,
    task, seed, test_size, stratify, n_jobs,
    out_csv, model_config_path=None
):
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
                out_dir=out_csv.parent,  # ✅ pass output directory for split saving
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"✅ ({job_index + 1}/{total_jobs}) Completed model={model_name} | encoding={encoding} | "
                  f"n_samples={n_samples} | ρ_test={result.get('rho_test', float('nan')):.4f} | "
                  f"time={elapsed:.2f}s | status={result.get('status', 'ok')}")
            return result
        except Exception as e:
            print(f"❌ ({job_index + 1}/{total_jobs}) Failed model={model_name} | encoding={encoding} | "
                  f"n_samples={n_samples} | error={e}")
            return {
                "model": model_name,
                "encoding": encoding,
                "n_samples": n_samples,
                "rho_train": float("nan"),
                "p_train": float("nan"),
                "rho_test": float("nan"),
                "p_test": float("nan"),
                "seconds": float("nan"),
                "status": f"error: {e}",
            }

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(run_and_log)(i, m, e, n) for i, (m, e, n) in enumerate(combos)
    )

    # Save progressively and track finished count
    for i, result in enumerate(parallel_results, 1):
        progress.append_to_csv(result, out_csv)
        print(f"[{i}/{total_jobs}] Saved {result['model']} | {result['encoding']} | n={result['n_samples']}")

    print(f"\n🏁 All {total_jobs} experiments finished.")
    return parallel_results


def _train_best_models(df_results, df, task, seed, out_dir, model_config_path=None):
    """
    Retrain best-performing models on their own training splits only,
    using the split files saved by run_single, and save the fitted Pipeline
    (StandardScaler(with_mean=False) -> model) into out_dir/models.
    """
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Prefer new metrics; fallback to legacy 'rho'
    metric_col = "rho_test" if "rho_test" in df_results.columns else "rho"
    print(f"🏗️  Selecting best models based on '{metric_col}'")

    for model_name, df_model in df_results.groupby("model"):
        try:
            best_row = df_model.sort_values(metric_col, ascending=False).iloc[0]
            encoding = best_row["encoding"]
            n_samples = int(best_row["n_samples"])

            # Split file saved by run_single for this exact combo
            split_file = out_dir / f"split_{model_name}_{encoding}_{n_samples}.csv"
            if not split_file.exists():
                print(f"⚠️  Split file missing for {model_name}/{encoding}/n={n_samples}: {split_file.name}. "
                      f"Skipping retrain to avoid data leakage.")
                continue

            # Load train/test partition captured during the winning run
            split_df = pd.read_csv(split_file)
            train_df = split_df[split_df["split"].str.lower() == "train"]
            if train_df.empty:
                print(f"⚠️  No training rows in {split_file.name}; skipping {model_name}.")
                continue

            # Re-encode sequences for training set only (matches run_single)
            X_train = encode_sequences(train_df["sequence"], encoding=encoding, k=3)
            y_train = train_df["label"].to_numpy()

            # Build model with same config; pack scaler to match run_single standardization
            model = build_model(task, model_name, seed=seed, model_config_path=model_config_path)
            pipeline = make_pipeline(StandardScaler(with_mean=False), model)

            print(f"🔁 Retraining best {model_name} on TRAIN split only "
                  f"({encoding}, n={n_samples}, train_n={len(train_df)})")
            pipeline.fit(X_train, y_train)

            # Name includes model, encoding, and sample size used in the BEST run
            model_path = models_dir / f"{model_name}_{encoding}_{n_samples}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"💾 Saved best {model_name} → {model_path.resolve()}")

        except Exception as e:
            print(f"⚠️  Could not retrain {model_name}: {type(e).__name__}: {e}")

    return models_dir

def _generate_report(out_dir, df_results, ranked, task, *_, **__):
    """
    Wrapper for backward compatibility with older calls.
    Calls report.generate_report(df_results, ranked, out_dir, task)
    and returns (plot_paths, meta) so that run_scout() and downstream
    code continue to work unchanged.
    """
    try:
        plot_paths, meta = report.generate_report(df_results, ranked, out_dir, task)
        return plot_paths, meta
    except Exception as e:
        print(f"⚠️  Report generation failed: {type(e).__name__}: {e}")
        return [], {"error": str(e)}


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
    outpath="model_scout_results",
    model_config_path=None,
):
    """Run the full Model Scout pipeline."""
    t0 = time.time()
    start_ts = datetime.now().isoformat(timespec="seconds")

    print("🔍 Loading data...")
    df = data_utils.load_data(sequences, labels)
    print(f"Loaded {len(df)} sequences with labels.")

    # --- Handle output directory ---
    out_dir = Path(outpath)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"
    out_csv = out_dir / "results.csv"
    out_csv.unlink(missing_ok=True)

    print(f"📁 Output directory: {out_dir.resolve()}")
    print(f"📊 Progressive results → {out_csv.resolve()}")

    # --- Main experiment run ---
    results = _run_parallel_jobs(
        df, models_list, encodings, sample_sizes, task,
        seed, test_size, stratify, n_jobs, out_csv,
        model_config_path=model_config_path
    )

    print("💾 Aggregating final results...")
    df_results = aggregator.save_results(results, out_json)
    ranked = aggregator.rank_results(df_results, alpha)

    print("\n🏆 Top-ranked combinations:")
    print(ranked.head(10).to_string(index=False))

    # --- Train best models + generate report ---
    models_dir = _train_best_models(df_results, df, task, seed, out_dir, model_config_path)
    plot_paths, meta = _generate_report(out_dir, df_results, ranked, task)

    # --- Total elapsed time ---
    total_seconds = round(time.time() - t0, 2)
    end_ts = datetime.now().isoformat(timespec="seconds")
    print(f"\n⏱️ Total elapsed time: {total_seconds}s (started {start_ts}, finished {end_ts})")

    # --- Final summary ---
    summary = {
        "out_json": str(out_json.resolve()),
        "out_csv": str(out_csv.resolve()),
        "models_dir": str(models_dir.resolve()),
        "plots": plot_paths,
        "meta": {
            **meta,
            "total_seconds": total_seconds,
            "start_time": start_ts,
            "end_time": end_ts,
        },
        "top": ranked.head(20).to_dict(orient="records"),
    }

    progress.write_summary(summary, out_json)
    print(f"\n✅ All done! Report and outputs saved in {out_dir.resolve()}")
    return summary
