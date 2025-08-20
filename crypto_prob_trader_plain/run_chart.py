
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config import settings
from datafeed import get_klines
from strategy import generate_signals
from features import make_features
from model import fit_prob_model, add_probabilities

WINDOW_BARS = 300
MAX_SL_PCT = 0.10
MAX_TP_PCT = 0.20

_model = None
_feats = None

def fetch():
    df = get_klines(settings.symbol, settings.interval, settings.limit)
    df, lvls = generate_signals(df, settings.rsi_len, settings.fib_lookback, settings.prox_pct)
    return df, lvls

def ensure_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    global _model, _feats
    feat_df, feats = make_features(df, rsi_len=settings.rsi_len,
                                   lookback=settings.fib_lookback,
                                   prox_pct=settings.prox_pct)
    if _model is None or _feats != feats:
        _model = fit_prob_model(feat_df, feats)
        _feats = feats
    out = add_probabilities(_model, feat_df, feats, thr=settings.threshold)
    return out

def update(frame, ax_price, ax_macd, ax_prob):
    ax_price.clear(); ax_macd.clear(); ax_prob.clear()
    ax_price.grid(True, alpha=0.25)
    ax_macd.grid(True, alpha=0.25)
    ax_prob.grid(True, alpha=0.25)

    df, lvls = fetch()
    use = df.tail(WINDOW_BARS).copy()
    x = pd.to_datetime(use.index)

    ax_price.plot(x, use["close"].values, label="Close", linewidth=1.1)
    if {"ema50","ema200"}.issubset(use.columns):
        ax_price.plot(x, use["ema50"].values, label="EMA50", linewidth=1.0)
        ax_price.plot(x, use["ema200"].values, label="EMA200", linewidth=1.0)

    for j, (name, lvl) in enumerate(lvls.items()):
        ax_price.axhline(lvl, linestyle="--", linewidth=0.8, alpha=0.5, label=name if j==0 else None)
        ax_price.text(x.iloc[-1], lvl, f"{lvl:.2f}", va="center", ha="left", fontsize=8)

    buys = use[use["signal"] == "BUY"]
    if not buys.empty:
        ax_price.scatter(pd.to_datetime(buys.index), buys["close"], marker="^", s=90, label="BUY")

    if not buys.empty and buys.index[-1] == use.index[-1]:
        price = use["close"].iloc[-1]
        stop_loss = price * (1 - MAX_SL_PCT)
        take_profit = price * (1 + MAX_TP_PCT)
        ax_price.axhline(stop_loss, linestyle="--")
        ax_price.text(x.iloc[-1], stop_loss, f"SL {stop_loss:.2f}", va="bottom", fontsize=8)
        ax_price.axhline(take_profit, linestyle="--")
        ax_price.text(x.iloc[-1], take_profit, f"TP {take_profit:.2f}", va="bottom", fontsize=8)

    last = use["close"].iloc[-1]
    ax_price.scatter(x.iloc[-1], last, zorder=5)
    ax_price.text(x.iloc[-1], last, f"{last:.2f}", va="bottom", ha="left", fontsize=9)

    ax_price.set_title(f"{settings.symbol} • {settings.interval} • Live")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", ncols=4, fontsize=8)

    if {"macd","macd_signal","macd_hist"}.issubset(use.columns):
        ax_macd.bar(x, use["macd_hist"].values, alpha=0.5)
        ax_macd.plot(x, use["macd"].values, label="MACD")
        ax_macd.plot(x, use["macd_signal"].values, label="Signal")
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(fontsize=8)

    prob_df = ensure_probabilities(df)
    prob_use = prob_df.tail(WINDOW_BARS)
    xp = pd.to_datetime(prob_use.index)
    ax_prob.plot(xp, prob_use["prob_up"].values, label="P(up)")
    ax_prob.axhline(settings.threshold, linestyle="--", alpha=0.5, label="BUY thr")
    ax_prob.axhline(1-settings.threshold, linestyle="--", alpha=0.5, label="SELL thr")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Probability")
    ax_prob.legend(fontsize=8)

    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax_prob.xaxis.get_majorticklabels(), rotation=15, ha='right')

def main():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1.2, 1]})
    ani = FuncAnimation(fig, update, fargs=(ax1, ax2, ax3),
                        interval=max(1000, settings.poll_seconds*1000))
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
