import numpy as np
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.metrics import r2_score, mean_squared_error
import warnings

def compute_metrics(task, y_true, y_pred, context=None):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", ConstantInputWarning)
        rho, p = spearmanr(y_true, y_pred)
    if any(issubclass(w.category, ConstantInputWarning) for w in wlist):
        msg = "[WARN] ConstantInputWarning"
        if context:
            msg += f" for model={context.get('model')} encoding={context.get('encoding')} n_samples={context.get('n_samples')}"
        print(msg)
        rho, p = 0.0, 1.0
    if np.isnan(rho): rho, p = 0.0, 1.0
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }
    return out