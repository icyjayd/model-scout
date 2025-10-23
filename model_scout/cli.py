import argparse
from . import config
from .runner import run_scout
from .utils.model_config_loader import export_default_configs

def cli_entry():
    parser = argparse.ArgumentParser(description="Benchmark multiple models and encodings on sequence-property data.")
    parser.add_argument("sequences", nargs="?", help="Path to sequences .csv or .npy file")
    parser.add_argument("--labels", help="Path to labels .csv or .npy file")
    parser.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    parser.add_argument("--encodings", nargs="+", default=config.DEFAULT_ENCODINGS)
    parser.add_argument("--samples", nargs="+", type=int, default=config.DEFAULT_SAMPLE_GRID)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--alpha", type=float, default=config.ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--stratify", action="store_true", help="Enable stratified sampling (classification only)")
    parser.add_argument("--jobs", type=int, default=config.N_JOBS)
    parser.add_argument("--out", default="model_scout_results.json")
    parser.add_argument("--model-config", default=None, help="Path to model_configs.json (optional)")
    parser.add_argument("--export-config", action="store_true", help="Export default model configuration and exit")

    args = parser.parse_args()

    if args.export_config:
        export_default_configs()
        return

    if not args.sequences or not args.labels:
        parser.error("sequences and --labels are required unless using --export-config")

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
        stratify=args.stratify,
        n_jobs=args.jobs,
        outpath=args.out,
        model_config_path=args.model_config,
    )

if __name__ == "__main__":
    cli_entry()
