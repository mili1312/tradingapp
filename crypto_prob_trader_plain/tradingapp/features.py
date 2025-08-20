
import numpy as np
import pandas as pd
from indicators import add_rsi, add_ema, add_atr, add_macd

def make_features(df: pd.DataFrame, rsi_len=14, lookback=200, prox_pct=0.25):
    out = df.copy()
    out = add_rsi(out, rsi_len)
    out = add_ema(out, spans=(50, 200))
    out = add_atr(out, length=14)
    out = add_macd(out, fast=12, slow=26, signal=9)

    out["ret"] = np.log(out["close"]).diff()
    out["vol_10"] = out["ret"].rolling(10).std()
    out["atr_norm"] = out["atr"] / out["close"]

    out["ema_spread"] = out["ema50"] - out["ema200"]
    out["ema_cross_up"]   = ((out["ema50"] > out["ema200"]) & (out["ema50"].shift(1) <= out["ema200"].shift(1))).astype(int)
    out["ema_cross_down"] = ((out["ema50"] < out["ema200"]) & (out["ema50"].shift(1) >= out["ema200"].shift(1))).astype(int)

    out["macd_edge"] = out["macd"] - out["macd_signal"]

    # Rolling Fibonacci (no look-ahead)
    hi = out["high"].rolling(lookback).max()
    lo = out["low"].rolling(lookback).min()
    rng = hi - lo
    fib382 = hi - 0.382 * rng
    fib50  = hi - 0.5   * rng
    fib618 = hi - 0.618 * rng

    d382 = (out["close"] - fib382).abs() / out["close"]
    d50  = (out["close"] - fib50 ).abs() / out["close"]
    d618 = (out["close"] - fib618).abs() / out["close"]
    out["fib_rel_dist"] = np.minimum.reduce([d382, d50, d618])
    out["near_fib"] = (out["fib_rel_dist"] <= (prox_pct/100.0)).astype(int)

    out["y"] = (out["ret"].shift(-1) > 0).astype(int)

    feature_cols = [
        "rsi", "ema_spread", "ema_cross_up", "ema_cross_down",
        "macd_edge", "fib_rel_dist", "near_fib", "vol_10", "atr_norm"
    ]
    out = out.dropna(subset=feature_cols + ["y"]).copy()
    return out, feature_cols
