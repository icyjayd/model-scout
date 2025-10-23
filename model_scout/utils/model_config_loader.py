"""
Loads per-model configuration dictionaries from model_configs.json.
Falls back to default hyperparameters if missing or incomplete.
"""

import json
from pathlib import Path

_DEFAULTS = {
    "ridge": {"alpha": 1.0, "fit_intercept": True, "solver": "auto"},
    "lasso": {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000},
    "enet": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000},
    "rf": {"n_estimators": 100, "max_depth": None, "random_state": 42, "n_jobs": -1},
    "gb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    "svr": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
    "mlp": {"hidden_layer_sizes": [100], "activation": "relu", "solver": "adam", "max_iter": 200, "random_state": 42},
    "xgb": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 42, "n_jobs": -1},
    "lgbm": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": -1, "random_state": 42, "n_jobs": -1},
}

def load_model_configs(config_path: str | None = None) -> dict:
    """Load model hyperparameter configs from JSON, falling back to defaults."""
    cfg_file = Path(config_path or "model_configs.json")
    if not cfg_file.exists():
        print(f"⚠️  No model_configs.json found at {cfg_file}. Using built-in defaults.")
        return _DEFAULTS

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        print(f"✅ Loaded model configs from {cfg_file}.")
    except Exception as e:
        print(f"⚠️  Failed to read {cfg_file}: {e}. Falling back to defaults.")
        return _DEFAULTS

    merged = {}
    for model_name, default_params in _DEFAULTS.items():
        user_params = user_cfg.get(model_name, {})
        merged[model_name] = {**default_params, **user_params}
    return merged

def export_default_configs(path: str = "model_configs.json") -> str:
    """Export the default configuration to a JSON file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_DEFAULTS, f, indent=2)
    print(f"✅ Exported default model configs to {path.resolve()}")
    return str(path.resolve())
