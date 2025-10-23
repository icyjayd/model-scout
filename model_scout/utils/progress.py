"""
Utilities for progressive result saving and final summary writing.
"""

import pandas as pd
import json
from pathlib import Path


def append_to_csv(result, out_csv: Path):
    """Append one result dictionary to a CSV file, creating it if missing."""
    df_row = pd.DataFrame([result])
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    df_row.to_csv(out_csv, mode="a", header=write_header, index=False)


def write_summary(summary: dict, out_json: Path):
    """Write a structured JSON summary file."""
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
