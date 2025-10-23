import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import project modules (ml_models.*) at runtime; main.py ensures sys.path includes project root.
from .encoding import encode_sequences
from .models import build_model

import warnings
from scipy.stats import spearmanr, ConstantInputWarning

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
        rho, p = 0.0, 1.0  # force safe defaults

    if np.isnan(rho): rho = 0.0
    if np.isnan(p): p = 1.0
    return float(rho), float(p)

def run_single(model_name, encoding, n_samples, df, task, seed, test_size, stratify):
    """Train and evaluate one model/encoding/sample-size combination."""
    try:
        df_sub = df.sample(n=min(n_samples, len(df)), random_state=seed)
        X = encode_sequences(df_sub["sequence"], encoding=encoding, k=3)
        y = df_sub["label"].values

        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
        strat_y = y if (task == "classification" and stratify == "auto") else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat_y
        )

        model = build_model(task, model_name)
        t0 = time.time()
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        seconds = round(time.time() - t0, 3)

        rho, p = _spearman(
            yte,
            ypred,
            context={"model": model_name, "encoding": encoding, "n_samples": n_samples},
        )
        return {
            "model": model_name, "encoding": encoding, "n_samples": int(n_samples),
            "rho": rho, "p": p, "seconds": seconds, "status": "ok"
        }
    except Exception as e:
        return {
            "model": model_name, "encoding": encoding, "n_samples": int(n_samples),
            "rho": 0.0, "p": 1.0, "seconds": 0,
            "status": f"error: {type(e).__name__}: {e}"
        }
