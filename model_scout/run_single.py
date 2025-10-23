import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

# Import project modules (ml_models.*) at runtime; main.py ensures sys.path includes project root.
from .encoding import encode_sequences
from .models import build_model


def _spearman(y_true, y_pred, context=None):
    """Compute Spearman ρ, catching constant-input warnings."""
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConstantInputWarning)
        rho, p = spearmanr(y_true, y_pred)

    if any(issubclass(w.category, ConstantInputWarning) for w in wlist):
        msg = "[WARN] ConstantInputWarning"
        if context:
            msg += f" for model={context.get('model')} encoding={context.get('encoding')} n_samples={context.get('n_samples')}"
        print(msg)
        rho, p = 0.0, 1.0  # safe defaults

    if np.isnan(rho): rho = 0.0
    if np.isnan(p): p = 1.0
    return float(rho), float(p)


def run_single(model_name, encoding, n_samples, df, task, seed, test_size, stratify, model_config_path=None, out_dir=None):
    """Train, evaluate, and save one model/encoding/sample-size combination with split reproducibility."""
    try:
        # --- Subsample the dataset ---
        df_sub = df.sample(n=min(n_samples, len(df)), random_state=seed)
        X = encode_sequences(df_sub["sequence"], encoding=encoding, k=3)
        y = df_sub["label"].values

        # --- Standardize ---
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

        # --- Stratification ---
        strat_y = y if (task == "classification" and stratify == "auto") else None

        # --- Split ---
        Xtr, Xte, ytr, yte, idx_train, idx_test = train_test_split(
            X, y, df_sub.index, test_size=test_size, random_state=seed, stratify=strat_y
        )

        # --- Model ---
        model = build_model(task, model_name, model_config_path=model_config_path, seed=seed)
        t0 = time.time()
        model.fit(Xtr, ytr)
        ypred_train = model.predict(Xtr)
        ypred_test = model.predict(Xte)
        seconds = round(time.time() - t0, 3)

        # --- Metrics ---
        rho_train, p_train = _spearman(
            ytr, ypred_train, context={"model": model_name, "encoding": encoding, "n_samples": n_samples}
        )
        rho_test, p_test = _spearman(
            yte, ypred_test, context={"model": model_name, "encoding": encoding, "n_samples": n_samples}
        )

        # --- Save split indices for reproducibility ---
        if out_dir:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            split_file = out_path / f"split_{model_name}_{encoding}_{n_samples}.csv"

            df_split = pd.DataFrame({
                "sequence": list(df_sub.loc[idx_train, "sequence"]) + list(df_sub.loc[idx_test, "sequence"]),
                "label": list(df_sub.loc[idx_train, "label"]) + list(df_sub.loc[idx_test, "label"]),
                "split": ["train"] * len(idx_train) + ["test"] * len(idx_test)
            })
            df_split.to_csv(split_file, index=False)
            print(f"💾 Saved split indices → {split_file.resolve()}")

        # --- Package results ---
        return {
            "model": model_name,
            "encoding": encoding,
            "n_samples": int(n_samples),
            "rho_train": rho_train,
            "p_train": p_train,
            "rho_test": rho_test,
            "p_test": p_test,
            "seconds": seconds,
            "status": "ok"
        }

    except Exception as e:
        return {
            "model": model_name,
            "encoding": encoding,
            "n_samples": int(n_samples),
            "rho_train": 0.0,
            "p_train": 1.0,
            "rho_test": 0.0,
            "p_test": 1.0,
            "seconds": 0.0,
            "status": f"error: {type(e).__name__}: {e}"
        }
