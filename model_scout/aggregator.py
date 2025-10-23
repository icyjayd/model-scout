import json
import pandas as pd
from pathlib import Path

def save_results(results, outpath):
    outdir = Path(outpath).parent
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    df = pd.DataFrame(results)
    df.to_csv(outdir / "model_scout_results.csv", index=False)
    return df

def rank_results(df: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """
    Rank results by best-performing metric.
    - For regression tasks: Spearman rho (higher is better)
    - For classification tasks: accuracy (higher is better, falls back to rho)
    """
    # Determine which metric to rank by
    metric_to_rank = "accuracy" if "accuracy" in df.columns and df["accuracy"].notna().any() else "rho"

    # Filter to significant results (based on p if available)
    if "p" in df.columns:
        df_sig = df[df["p"] <= alpha]
        if df_sig.empty:
            df_sig = df
    else:
        df_sig = df

    # Compute group-wise best metrics
    agg_dict = {metric_to_rank: "max", "n_samples": "min"}
    if "p" in df_sig.columns:
        agg_dict["p"] = "min"

    grouped = (
        df_sig.groupby(["model", "encoding"], as_index=False)
        .agg(agg_dict)
        .sort_values(metric_to_rank, ascending=False)
        .reset_index(drop=True)
    )

    grouped.rename(columns={metric_to_rank: "best_metric"}, inplace=True)
    grouped["metric_type"] = metric_to_rank

    return grouped
