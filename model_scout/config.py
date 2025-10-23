from joblib import cpu_count

DEFAULT_MODELS = [
    "ridge", "lasso", "enet", "rf", "gb", "xgb", "lgbm", "svr", "mlp",
]
DEFAULT_ENCODINGS = ["aac", "dpc", "kmer", "onehot"]
DEFAULT_SAMPLE_GRID = [2000, 5000, 10000]
ALPHA = 0.01
N_JOBS = max(cpu_count() - 1, 1)
