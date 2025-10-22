import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .metrics import compute_metrics

def train_model(X, y, model, task="regression", test_size=0.2, seed=42, context=None):
    idx = np.arange(len(y))
    stratify_y = y if (task == "classification" and len(set(y)) > 1) else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify_y)
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(task, y_test, y_pred, context=context)