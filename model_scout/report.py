import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(df_results, ranked, out_dir, task):
    """Generate simple HTML and plots summarizing train/test performance."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary Table ---
    summary_html = out_dir / "summary.html"
    df_display = ranked.copy()

    # Compute overfitting indicator
    if "rho_train" in df_display.columns and "rho_test" in df_display.columns:
        df_display["Δρ"] = df_display["rho_train"] - df_display["rho_test"]
        df_display["overfit_flag"] = df_display["Δρ"].apply(
            lambda x: "⚠️" if x > 0.1 else ""
        )

    df_display_html = df_display.to_html(index=False, float_format="%.4f")

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Model Scout Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            th {{ background-color: #eee; }}
        </style>
    </head>
    <body>
        <h1>Model Scout Results — {task.title()}</h1>
        <p>Showing Spearman ρ and p-values for training and test sets.</p>
        {df_display_html}
    </body>
    </html>
    """

    with open(summary_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📊 Generated summary → {summary_html.resolve()}")

    # --- Correlation Comparison Plot ---
    if {"rho_train", "rho_test"}.issubset(df_results.columns):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df_results,
            x="rho_train",
            y="rho_test",
            hue="encoding",
            style="model",
            s=100
        )
        plt.title("Train vs Test Spearman Correlation")
        plt.xlabel("ρ_train")
        plt.ylabel("ρ_test")
        plt.grid(True)
        plt.tight_layout()

        plot_path = out_dir / "train_vs_test_rho.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"📈 Saved train/test plot → {plot_path.resolve()}")

    return [summary_html], {"n_results": len(df_results), "top_rho": df_display["rho_test"].max()}
