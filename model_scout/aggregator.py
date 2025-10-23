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

def rank_results(df, alpha=0.01):
    df_valid = df[df["p"] <= alpha].copy()
    if len(df_valid) == 0:
        df_valid = df.copy()
    grouped = (
        df_valid.groupby(["model", "encoding"], as_index=False)
        .agg({"rho": "max", "n_samples": "min", "p": "min"})
        .sort_values("rho", ascending=False)
    )
    return grouped
