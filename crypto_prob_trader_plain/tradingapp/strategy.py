import pandas as pd
from indicators import add_rsi, add_ema, add_atr, add_macd, fib_levels, near_any_fib
from features import make_features
from model import fit_prob_model, add_probabilities
from config import settings

# Conservative strategy
def generate_signals(df: pd.DataFrame,
                     rsi_len=14,
                     lookback=200,
                     prox_pct=0.25,
                     prob_thr=None):

    # --- Indicators ---
    df = add_rsi(df, rsi_len).copy()
    df = add_ema(df, spans=(50, 200))
    df = add_atr(df, length=14)
    df = add_macd(df, fast=12, slow=26, signal=9)

    # --- Fibonacci ---
    lvls = fib_levels(df, lookback)

    # --- Probabilities ---
    feat_df, feats = make_features(df, rsi_len=rsi_len,
                                   lookback=lookback,
                                   prox_pct=prox_pct)
    model = fit_prob_model(feat_df, feats)
    prob_df = add_probabilities(model, feat_df, feats,
                                thr=prob_thr or settings.threshold)
    df["prob_up"] = prob_df["prob_up"]

    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        s = ""
        price = row["close"]

        if i > 0:
            prev = df.iloc[i - 1]
            golden_cross = (row["ema50"] > row["ema200"]) and (prev["ema50"] <= prev["ema200"])
            death_cross  = (row["ema50"] < row["ema200"]) and (prev["ema50"] >= prev["ema200"])
        else:
            golden_cross = death_cross = False

        # --- Filters ---
        rsi = row.get("rsi")
        near = near_any_fib(price, lvls, prox_pct)
        macd_ok = row["macd"] > row["macd_signal"]
        prob_up = row.get("prob_up", 0.5)

        # --- BUY rules ---
        if golden_cross and rsi < 50 and near and macd_ok and prob_up > (prob_thr or settings.threshold):
            s = "BUY"

        # --- SELL rules ---
        elif death_cross and rsi > 70 and near and not macd_ok and prob_up < (1 - (prob_thr or settings.threshold)):
            s = "SELL"

        signals.append(s)

    df["signal"] = signals
    return df, lvls
