from .utils.model_config_loader import load_model_configs
import inspect
from copy import deepcopy

# Models that accept random_state safely
_RANDOM_STATE_MODELS = {"rf", "gb", "mlp", "xgb", "lgbm", "ridge"}


def _filter_valid_params(cls, params):
    """Remove any kwargs not accepted by the model's __init__."""
    valid = set(inspect.signature(cls).parameters.keys())
    return {k: v for k, v in params.items() if k in valid}


def build_model(task, model_name, seed=42, model_config_path=None):
    """
    Build a model instance using parameters from model_configs.json.
    Supports both regression and classification variants.
    """
    all_cfg = load_model_configs(model_config_path)
    params = deepcopy(all_cfg.get(model_name, {}))

    # Only inject random_state for models that support it
    if "random_state" not in params and model_name in _RANDOM_STATE_MODELS:
        params["random_state"] = seed

    # --- LINEAR MODELS ---
    if model_name == "ridge":
        from sklearn.linear_model import Ridge, RidgeClassifier
        if task == "classification":
            return RidgeClassifier(**_filter_valid_params(RidgeClassifier, params))
        return Ridge(**_filter_valid_params(Ridge, params))

    elif model_name == "lasso":
        from sklearn.linear_model import Lasso, LogisticRegression
        if task == "classification":
            # LogisticRegression with L1 penalty as a classifier analog to Lasso
            clf_params = deepcopy(params)
            clf_params.pop("max_iter", None)
            return LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=1000,
                **_filter_valid_params(LogisticRegression, clf_params)
            )
        return Lasso(**_filter_valid_params(Lasso, params))

    elif model_name == "enet":
        from sklearn.linear_model import ElasticNet, LogisticRegression
        if task == "classification":
            # LogisticRegression with elasticnet penalty for classification
            clf_params = deepcopy(params)
            clf_params.pop("max_iter", None)
            l1_ratio = clf_params.pop("l1_ratio", 0.5)
            return LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                max_iter=1000,
                **_filter_valid_params(LogisticRegression, clf_params)
            )
        return ElasticNet(**_filter_valid_params(ElasticNet, params))

    # --- TREE ENSEMBLES ---
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(**_filter_valid_params(cls, params))

    elif model_name == "gb":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(**_filter_valid_params(cls, params))

    # --- SVM ---
    elif model_name == "svr":
        from sklearn.svm import SVR, SVC
        if task == "classification":
            safe_params = deepcopy(params)
            for bad_key in ["epsilon", "random_state"]:
                safe_params.pop(bad_key, None)
            return SVC(**_filter_valid_params(SVC, safe_params))
        return SVR(**_filter_valid_params(SVR, params))

    # --- NEURAL NET ---
    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if task == "classification" else MLPRegressor
        return cls(**_filter_valid_params(cls, params))

    # --- BOOSTING LIBRARIES ---
    elif model_name == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(**_filter_valid_params(cls, params))

    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if task == "classification" else LGBMRegressor
        return cls(**_filter_valid_params(cls, params))

    else:
        raise ValueError(f"Unknown model type: {model_name}")
