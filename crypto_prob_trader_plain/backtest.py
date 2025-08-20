
import pandas as pd
import numpy as np
from config import settings
from datafeed import get_klines
from strategy import generate_signals
from features import make_features
from model import fit_prob_model, add_probabilities, evaluate

def simple_long_only(df: pd.DataFrame, fee_rate: float = 0.0004, sl_pct: float = None, tp_pct: float = None):
    balance = 0.0
    equity_curve = []
    in_pos = False
    entry = 0.0
    max_dd = 0.0
    peak = 0.0
    trades = []
    for ts, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]
        if not in_pos and signal == "BUY":
            in_pos = True
            entry = price
            trades.append({"time_in": ts, "entry": entry})
        elif in_pos:
            hit_exit = False
            exit_price = price
            if sl_pct is not None and row["low"] <= entry * (1 - sl_pct/100):
                exit_price = entry * (1 - sl_pct/100); hit_exit = True
            if not hit_exit and tp_pct is not None and row["high"] >= entry * (1 + tp_pct/100):
                exit_price = entry * (1 + tp_pct/100); hit_exit = True
            if signal == "SELL" or hit_exit:
                ret = (exit_price * (1 - fee_rate)) / (entry * (1 + fee_rate)) - 1
                balance += ret
                trades[-1].update({"time_out": ts, "exit": exit_price, "ret": ret})
                in_pos = False
        peak = max(peak, balance)
        dd = peak - balance
        max_dd = max(max_dd, dd)
        equity_curve.append(balance)
    res = {
        "total_return": balance,
        "max_drawdown": max_dd,
        "equity_curve": pd.Series(equity_curve, index=df.index),
        "trades": pd.DataFrame(trades)
    }
    return res

def run_backtests():
    print(f"Fetching data {settings.symbol} {settings.interval} {settings.limit}...")
    df_raw = get_klines(settings.symbol, settings.interval, settings.limit)

    rule_df, _ = generate_signals(df_raw, settings.rsi_len, settings.fib_lookback, settings.prox_pct)

    feat_df, feats = make_features(df_raw, settings.rsi_len, settings.fib_lookback, settings.prox_pct)
    model = fit_prob_model(feat_df, feats)
    print("Prob model train metrics:", evaluate(model, feat_df, feats))
    prob_df = add_probabilities(model, feat_df, feats, thr=settings.threshold)

    merged = df_raw.join(rule_df[["signal"]], how="left").join(prob_df[["prob_up","signal_prob"]], how="left")
    merged["signal_rule"] = merged["signal"].fillna("")
    merged["signal_prob"] = merged["signal_prob"].fillna("")
    merged["signal_hybrid"] = np.where((merged["signal_rule"]=="BUY") & (merged["signal_prob"]=="BUY"), "BUY",
                                np.where((merged["signal_rule"]=="SELL") & (merged["signal_prob"]=="SELL"), "SELL", ""))

    print("\nRunning backtests (fee=0.04%)...\n")
    res_rule  = simple_long_only(merged.assign(signal=merged["signal_rule"]),  fee_rate=0.0004, sl_pct=1.0, tp_pct=2.0)
    res_prob  = simple_long_only(merged.assign(signal=merged["signal_prob"]),  fee_rate=0.0004, sl_pct=1.0, tp_pct=2.0)
    res_hyb   = simple_long_only(merged.assign(signal=merged["signal_hybrid"]),fee_rate=0.0004, sl_pct=1.0, tp_pct=2.0)

    def summarize(name, res):
        total = res["total_return"]
        dd = res["max_drawdown"]
        trades = len(res["trades"]) if not res["trades"].empty else 0
        print(f"{name:10s} | total={total:.3f} | maxDD={dd:.3f} | trades={trades}")

    summarize("Rule", res_rule)
    summarize("Prob", res_prob)
    summarize("Hybrid", res_hyb)

    return {"rule": res_rule, "prob": res_prob, "hybrid": res_hyb, "merged": merged}

if __name__ == "__main__":
    run_backtests()
