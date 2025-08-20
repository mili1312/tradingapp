
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

def time_split(df, train_ratio=0.7):
    i = int(len(df)*train_ratio)
    return df.iloc[:i].copy(), df.iloc[i:].copy()

def fit_prob_model(df, feature_cols, C=1.0):
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    X, y = df[feature_cols].values, df["y"].values
    model.fit(X, y)
    return model

def evaluate(model, df, feature_cols):
    X, y = df[feature_cols].values, df["y"].values
    p = model.predict_proba(X)[:,1]
    return {"AUC": float(roc_auc_score(y,p)),
            "Brier": float(brier_score_loss(y,p)),
            "LogLoss": float(log_loss(y,p))}

def add_probabilities(model, df, feature_cols, thr=0.55):
    X = df[feature_cols].values
    out = df.copy()
    out["prob_up"] = model.predict_proba(X)[:,1]
    out["signal_prob"] = np.where(out["prob_up"] > thr, "BUY",
                          np.where(out["prob_up"] < 1-thr, "SELL", ""))
    return out
