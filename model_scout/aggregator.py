import json
import pandas as pd
from pathlib import Path

def save_results(results, out_json):
    """Save all run results to both JSON and CSV formats."""
    df = pd.DataFrame(results)

    # Ensure all expected columns exist even if missing from some runs
    for col in [
        "model", "encoding", "n_samples",
        "rho_train", "p_train", "rho_test", "p_test",
        "seconds", "status"
    ]:
        if col not in df.columns:
            df[col] = None

    df.to_json(out_json, orient="records", indent=2)
    csv_path = Path(out_json).with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    print(f"💾 Saved results JSON → {out_json.resolve()}")
    print(f"💾 Saved results CSV  → {csv_path.resolve()}")

    return df


def rank_results(df, alpha=0.05):
    """
    Rank results by rho_test descending (primary metric),
    then by runtime ascending, then by rho_train for tie-breaking.
    """
    # Only keep successful runs
    df_ok = df[df["status"].str.lower().eq("ok")]

    if "rho_test" in df_ok.columns:
        df_ranked = df_ok.sort_values(
            by=["rho_test", "rho_train", "seconds"],
            ascending=[False, False, True],
            na_position="last"
        ).reset_index(drop=True)
    else:
        # Fallback to old metric naming if necessary
        df_ranked = df_ok.sort_values(
            by=["rho", "seconds"],
            ascending=[False, True],
            na_position="last"
        ).reset_index(drop=True)

    # Optional filtering by significance
    if "p_test" in df_ranked.columns:
        df_ranked = df_ranked[df_ranked["p_test"] < alpha].reset_index(drop=True)

    return df_ranked
