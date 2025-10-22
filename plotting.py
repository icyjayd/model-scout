import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

def make_plots(df_results, outdir):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")

    # Heatmap
    heat = (
        df_results.groupby(["model", "encoding"])["rho"]
        .max().unstack(fill_value=0).sort_index()
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Best Spearman ρ per model/encoding")
    plt.tight_layout()
    heatmap_path = os.path.join(outdir, "heatmap_rho.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results, x="n_samples", y="rho",
        hue="model", style="encoding", markers=True, dashes=False
    )
    plt.title("Spearman ρ vs sample size")
    plt.tight_layout()
    rho_vs_samples_path = os.path.join(outdir, "rho_vs_samples.png")
    plt.savefig(rho_vs_samples_path)
    plt.close()

    # Runtime bar plot
    plt.figure(figsize=(10, 6))
    rt = df_results.groupby("model", as_index=False)["seconds"].mean()
    sns.barplot(data=rt, x="model", y="seconds", palette="crest")
    plt.title("Average runtime per model (seconds)")
    plt.tight_layout()
    runtime_path = os.path.join(outdir, "runtime_per_model.png")
    plt.savefig(runtime_path)
    plt.close()

    return {
        "heatmap": heatmap_path,
        "rho_vs_samples": rho_vs_samples_path,
        "runtime": runtime_path
    }
