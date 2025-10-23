"""
Full coverage tests for Model Scout.
Ensures all model types and encodings run correctly for both regression and classification,
and verifies that progressive CSV generation produces valid, complete results.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from model_scout import run_single, config, aggregator


# --- Synthetic data ---
SEQUENCES = [
    "MKTIIALSYIFCLVFAD", "GASERPLVY", "MKLFWLLFTIGFCWA", "GLSDGEWQQVLNVWGK",
    "MGDVEKGKKIFIMKCSQ", "MATNRQLER", "MAVMAPRTLVLLLSGAL", "MQIFVKTLTGKTITLEV",
    "MTEITAAMVKELRESTG", "MADQLTEEQIAEFKEAF"
]
LABELS_REG = np.linspace(0, 1, len(SEQUENCES))
LABELS_CLS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
SAMPLE_SIZE = 5


# --------------------------
# Regression coverage
# --------------------------

@pytest.mark.parametrize("model_name", config.DEFAULT_MODELS)
@pytest.mark.parametrize("encoding_name", config.DEFAULT_ENCODINGS)
def test_all_models_encodings_regression(model_name, encoding_name):
    """Test all models and encodings with regression task."""

    result = run_single.run_single(
        sequences=SEQUENCES,
        labels=LABELS_REG,
        model_name=model_name,
        encoding=encoding_name,
        n_samples=SAMPLE_SIZE,
        task="regression",
        seed=42,
        test_size=0.2,
    )

    # Basic structure
    assert isinstance(result, dict)
    expected_keys = ["model", "encoding", "n_samples", "rho", "p", "seconds", "status"]
    for key in expected_keys:
        assert key in result, f"Missing key {key} for {model_name}/{encoding_name}"

    # Values should be numeric and finite
    for num_key in ["rho", "p", "seconds"]:
        val = result[num_key]
        assert isinstance(val, (float, int)), f"{num_key} is not numeric"
        assert np.isfinite(val), f"{num_key} not finite"

    # No failure message
    assert "error" not in result["status"].lower(), f"Run failed: {result['status']}"


# --------------------------
# Classification coverage
# --------------------------

@pytest.mark.parametrize("model_name", config.DEFAULT_MODELS)
@pytest.mark.parametrize("encoding_name", config.DEFAULT_ENCODINGS)
def test_all_models_encodings_classification(model_name, encoding_name):
    """Test all models and encodings with classification task."""

    result = run_single.run_single(
        sequences=SEQUENCES,
        labels=LABELS_CLS,
        model_name=model_name,
        encoding=encoding_name,
        n_samples=SAMPLE_SIZE,
        task="classification",
        seed=42,
        test_size=0.2,
    )

    # Structure & validity checks
    assert isinstance(result, dict)
    for key in ["model", "encoding", "n_samples", "rho", "p", "seconds", "status"]:
        assert key in result
    for k in ["rho", "p", "seconds"]:
        assert np.isfinite(result[k])
    assert "error" not in result["status"].lower()


# --------------------------
# CSV generation test
# --------------------------

def test_csv_generation_and_integrity(tmp_path):
    """
    Simulate progressive result saving and check that CSV
    contains correct columns and non-null values.
    """
    csv_path = tmp_path / "results.csv"
    results = []

    # Generate one small result per model/encoding
    for model_name in config.DEFAULT_MODELS:
        for encoding_name in config.DEFAULT_ENCODINGS:
            result = run_single.run_single(
                sequences=SEQUENCES,
                labels=LABELS_REG,
                model_name=model_name,
                encoding=encoding_name,
                n_samples=SAMPLE_SIZE,
                task="regression",
                seed=42,
                test_size=0.2,
            )
            results.append(result)
            # Append progressively
            df_row = pd.DataFrame([result])
            df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    # Validate CSV content
    assert csv_path.exists(), "CSV file not created"
    df = pd.read_csv(csv_path)

    # Required columns
    required_cols = ["model", "encoding", "n_samples", "rho", "p", "seconds", "status"]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col}"

    # No NaN or empty values
    assert not df.isna().any().any(), "CSV contains NaN values"

    # Reasonable types and ranges
    assert (df["rho"].apply(np.isfinite)).all(), "Invalid rho values"
    assert (df["p"].apply(np.isfinite)).all(), "Invalid p values"
    assert (df["seconds"] >= 0).all(), "Negative runtime detected"

    # Save aggregated form using aggregator to ensure consistency
    out_json = tmp_path / "out.json"
    df_final = aggregator.save_results(results, out_json)
    assert not df_final.empty
    assert set(required_cols).issubset(df_final.columns)
    
# --------------------------
# CLI smoke test
# --------------------------

def test_cli_entrypoint(tmp_path):
    """
    Smoke test for the CLI entrypoint.
    Runs one model/encoding/sample size end-to-end and checks that
    JSON, CSV, and model outputs are generated without error.
    """
    import sys
    import pandas as pd
    from model_scout import main

    seq_path = tmp_path / "seqs.csv"
    lab_path = tmp_path / "labels.csv"
    out_json = tmp_path / "out.json"

    # Create dummy dataset
    pd.DataFrame({"sequence": SEQUENCES}).to_csv(seq_path, index=False)
    pd.DataFrame({"label": LABELS_REG}).to_csv(lab_path, index=False)

    # Build CLI args list
    cli_args = [
        str(seq_path),
        "--labels", str(lab_path),
        "--models", "ridge",
        "--encodings", "aac",
        "--samples", "5",
        "--out", str(out_json),
    ]

    # Simulate CLI call
    sys.argv = ["model-scout"] + cli_args
    main.cli_entry()

    # Check that main outputs were created
    csv_path = out_json.with_suffix(".csv")
    models_dir = tmp_path / "models"

    assert out_json.exists(), "Output JSON missing after CLI run"
    assert csv_path.exists(), "Progressive CSV missing after CLI run"
    assert models_dir.exists(), "Models directory not created"

    # Verify JSON has expected structure
    import json
    data = json.loads(out_json.read_text())
    assert "out_csv" in data and "models_dir" in data

