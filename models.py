from sklearn.linear_model import LogisticRegression,Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor

try: import xgboost as xgb; HAS_XGB=True
except: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGBM=True
except: HAS_LGBM=False

def build_model(task,name,seed=42):
    n=name.lower()
    if task=="classification":
        if n in {"logreg","logistic"}: return LogisticRegression(max_iter=500,random_state=seed)
        if n in {"svm","svm_rbf"}: return SVC(kernel="rbf",probability=True,random_state=seed)
        if n in {"rf","random_forest"}: return RandomForestClassifier(n_estimators=400,random_state=seed)
        if n in {"gb","gboost"}: return GradientBoostingClassifier(random_state=seed)
        if n in {"mlp"}: return MLPClassifier(hidden_layer_sizes=(512,128),max_iter=50,random_state=seed)
        if n in {"xgb","xgboost"} and HAS_XGB: return xgb.XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=seed)
        if n in {"lgbm","lightgbm"} and HAS_LGBM: return lgb.LGBMClassifier(random_state=seed)
    else:
        if n in {"linreg","linear"}: return LinearRegression()
        if n in {"ridge"}: return Ridge(random_state=seed)
        if n in {"lasso"}: return Lasso(random_state=seed)
        if n in {"enet","elasticnet"}: return ElasticNet(random_state=seed)
        if n in {"svr"}: return SVR()
        if n in {"rf","random_forest"}: return RandomForestRegressor(random_state=seed)
        if n in {"gb","gboost"}: return GradientBoostingRegressor(random_state=seed)
        if n in {"mlp"}: return MLPRegressor(hidden_layer_sizes=(512,128),max_iter=100,random_state=seed)
        if n in {"xgb","xgboost"} and HAS_XGB: return xgb.XGBRegressor(random_state=seed)
        if n in {"lgbm","lightgbm"} and HAS_LGBM: return lgb.LGBMRegressor(random_state=seed)
    raise ValueError(f"Unsupported model {name}")
