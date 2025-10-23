"""
Full coverage tests for Model Scout.
Covers:
 - All model types × encodings (regression + classification)
 - CSV generation and aggregator integrity
 - CLI entrypoint producing JSON, CSV, models, and HTML report
"""

import pytest
import numpy as np
import pandas as pd
import sys
import json
import importlib
from pathlib import Path

from model_scout import run_single, config, aggregator, main


# --- Synthetic data ---
SEQUENCES = [
    "MKTIIALSYIFCLVFAD", "GASERPLVY", "MKLFWLLFTIGFCWA", "GLSDGEWQQVLNVWGK",
    "MGDVEKGKKIFIMKCSQ", "MATNRQLER", "MAVMAPRTLVLLLSGAL", "MQIFVKTLTGKTITLEV",
    "MTEITAAMVKELRESTG", "MADQLTEEQIAEFKEAF"
]
LABELS_REG = np.linspace(0, 1, len(SEQUENCES))
LABELS_CLS = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
SAMPLE_SIZE = 5


# --- Skip optional libraries if missing ---
def has_lib(name):
    return importlib.util.find_spec(name) is not None


# --------------------------
# Regression coverage
# --------------------------

@pytest.mark.parametrize("model_name", config.DEFAULT_MODELS)
@pytest.mark.parametrize("encoding_name", config.DEFAULT_ENCODINGS)
def test_all_models_encodings_regression(model_name, encoding_name):
    """Test all (model, encoding) pairs for regression."""

    if model_name == "xgb" and not has_lib("xgboost"):
        pytest.skip("xgboost not installed")
    if model_name == "lgbm" and not has_lib("lightgbm"):
        pytest.skip("lightgbm not installed")

    df = pd.DataFrame({"sequence": SEQUENCES, "label": LABELS_REG})

    result = run_single.run_single(
        model_name=model_name,
        encoding=encoding_name,
        n_samples=SAMPLE_SIZE,
        df=df,
        task="regression",
        seed=42,
        test_size=0.2,
        stratify=False,
    )

    assert isinstance(result, dict)
    for key in ["model", "encoding", "n_samples", "rho", "p", "seconds", "status"]:
        assert key in result
    for num_key in ["rho", "p", "seconds"]:
        val = result[num_key]
        assert isinstance(val, (float, int))
        assert np.isfinite(val)
    assert "error" not in result["status"].lower()


# --------------------------
# Classification coverage
# --------------------------

@pytest.mark.parametrize("model_name", config.DEFAULT_MODELS)
@pytest.mark.parametrize("encoding_name", config.DEFAULT_ENCODINGS)
def test_all_models_encodings_classification(model_name, encoding_name):
    """Test all (model, encoding) pairs for classification."""
    if model_name == "xgb" and not has_lib("xgboost"):
        pytest.skip("xgboost not installed")
    if model_name == "lgbm" and not has_lib("lightgbm"):
        pytest.skip("lightgbm not installed")

    df = pd.DataFrame({"sequence": SEQUENCES, "label": LABELS_CLS})

    result = run_single.run_single(
        model_name=model_name,
        encoding=encoding_name,
        n_samples=SAMPLE_SIZE,
        df=df,
        task="classification",
        seed=42,
        test_size=0.2,
        stratify=True,
    )

    assert isinstance(result, dict)
    assert "error" not in result["status"].lower(), f"{model_name} failed: {result['status']}"
    for k in ["rho", "p", "seconds"]:
        assert np.isfinite(result[k]), f"{model_name}-{encoding_name}: invalid {k}"

# --------------------------
# CSV generation test
# --------------------------

def test_csv_generation_and_integrity(tmp_path):
    """Check that CSV and aggregator outputs are valid."""
    csv_path = tmp_path / "results.csv"
    results = []

    df = pd.DataFrame({"sequence": SEQUENCES, "label": LABELS_REG})

    for model_name in config.DEFAULT_MODELS:
        for encoding_name in config.DEFAULT_ENCODINGS:
            if model_name == "xgb" and not has_lib("xgboost"):
                continue
            if model_name == "lgbm" and not has_lib("lightgbm"):
                continue

            result = run_single.run_single(
                model_name=model_name,
                encoding=encoding_name,
                n_samples=SAMPLE_SIZE,
                df=df,
                task="regression",
                seed=42,
                test_size=0.2,
                stratify=False,
            )
            results.append(result)
            pd.DataFrame([result]).to_csv(
                csv_path, mode="a", header=not csv_path.exists(), index=False
            )

    df_csv = pd.read_csv(csv_path)
    required_cols = ["model", "encoding", "n_samples", "rho", "p", "seconds", "status"]
    for col in required_cols:
        assert col in df_csv.columns
    assert not df_csv.isna().any().any()

    out_json = tmp_path / "out.json"
    df_final = aggregator.save_results(results, out_json)
    assert not df_final.empty


# --------------------------
# CLI smoke test
# --------------------------

def test_cli_entrypoint(tmp_path):
    """Smoke test for CLI: verifies JSON, CSV, models folder, and HTML report creation."""
    seq_path = tmp_path / "seqs.csv"
    lab_path = tmp_path / "labels.csv"
    out_json = tmp_path / "out.json"

    pd.DataFrame({"id": range(len(SEQUENCES)), "sequence": SEQUENCES}).to_csv(seq_path, index=False)
    pd.DataFrame({"id": range(len(SEQUENCES)), "label": LABELS_REG}).to_csv(lab_path, index=False)

    sys.argv = [
        "model-scout",
        str(seq_path),
        "--labels", str(lab_path),
        "--models", "ridge",
        "--encodings", "aac",
        "--samples", "5",
        "--out", str(out_json),
    ]
    main.cli_entry()

    csv_path = out_json.with_suffix(".csv")
    models_dir = tmp_path / "models"
    html_report = tmp_path / "reports" / "summary.html"

    assert out_json.exists()
    assert csv_path.exists()
    assert models_dir.exists()
    assert html_report.exists()

    data = json.loads(out_json.read_text())
    for key in ["out_csv", "models_dir", "meta", "plots", "top"]:
        assert key in data
