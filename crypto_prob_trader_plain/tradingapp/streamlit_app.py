import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from datafeed import get_klines
from strategy import generate_signals
from features import make_features
from model import fit_prob_model, add_probabilities
from ws_live import LiveTicker

st.set_page_config(page_title="Crypto Prob Trader", layout="wide")

# ---------- Helpers ----------
def interval_to_timedelta(interval_str: str) -> pd.Timedelta:
    u = interval_str[-1].lower(); n = int(interval_str[:-1])
    return {"m": pd.Timedelta(minutes=n),
            "h": pd.Timedelta(hours=n),
            "d": pd.Timedelta(days=n)}[u]

@st.cache_resource
def get_model_and_feats(df):
    feat_df, feats = make_features(
        df, rsi_len=settings.rsi_len,
        lookback=settings.fib_lookback,
        prox_pct=settings.prox_pct
    )
    model = fit_prob_model(feat_df, feats)
    return model, feats

@st.cache_resource
def start_ws(symbol: str):
    lt = LiveTicker(symbol)
    lt.start()
    return lt

# ---------- Sidebar ----------
st.sidebar.title("Settings")
poll = st.sidebar.number_input("Auto-refresh (sec)", 5, 300, value=int(settings.poll_seconds))
thr  = st.sidebar.slider("Prob threshold", 0.50, 0.80, value=float(settings.threshold), step=0.01)
symbol = st.sidebar.text_input("Symbol", settings.symbol)
intervals = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"]
default_idx = intervals.index(settings.interval) if settings.interval in intervals else intervals.index("1h")
interval = st.sidebar.selectbox("Interval", intervals, index=default_idx)
if st.sidebar.button("Refresh now"):
    st.rerun()

st.title(f"📈 Crypto Prob Trader — {symbol} {interval}")

# ---------- Fetch logic (κάθε poll sec μόνο) ----------
now = time.time()
store_key = f"store::{symbol}::{interval}"
if store_key not in st.session_state:
    st.session_state[store_key] = {"last_fetch": 0, "df": None, "lvls": None, "model": None, "feats": None}

store = st.session_state[store_key]
need_fetch = (now - store["last_fetch"] >= poll) or store["df"] is None

try:
    if need_fetch:
        raw = get_klines(symbol, interval, settings.limit)
        df_ind, lvls = generate_signals(raw, settings.rsi_len, settings.fib_lookback, settings.prox_pct)

        # probabilities
        model, feats = get_model_and_feats(raw)
        feat_df, _ = make_features(raw, settings.rsi_len, settings.fib_lookback, settings.prox_pct)
        prob_df = add_probabilities(model, feat_df, feats, thr=thr)
        df_ind = df_ind.drop(columns=["prob_up","signal_prob"], errors="ignore").join(
            prob_df[["prob_up","signal_prob"]], how="left"
        )

        store.update({"df": df_ind, "lvls": lvls, "last_fetch": now, "model": model, "feats": feats})

    # Πάρε τα τρέχοντα από το store (μένουν ορατά συνέχεια)
    df = store["df"]
    lvls = store["lvls"]

    # --- Header metrics ---
    last = df.iloc[-1]
    ts_open = df.index[-1]
    bar_close = (last["close_time"] if "close_time" in df.columns
                 else ts_open + interval_to_timedelta(interval))
    if not isinstance(bar_close, pd.Timestamp):
        bar_close = pd.to_datetime(bar_close, utc=True)
    elif bar_close.tzinfo is None:
        bar_close = bar_close.tz_localize("UTC")
    eta = max(pd.Timedelta(0), bar_close - pd.Timestamp.now(tz="UTC"))

    # WebSocket live
    lt = start_ws(symbol)

    # --- Top row (placeholders) ---
    c0, c1, c2, c3 = st.columns(4)
    price_ph = c0.empty()
    c1.metric("Prob(up)", f"{last.get('prob_up', float('nan')):.2f}")
    c2.metric("RSI", f"{last.get('rsi', float('nan')):.2f}")
    c3.metric("ETA to close", str(eta).split(".")[0])

    # --- Price chart (always visible) ---
    chart_ph = st.empty()

    def render_chart(live_price_val: float):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(width=1.2)))
        if "ema50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50"))
        if "ema200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema200"], name="EMA200"))
        for name, lvl in lvls.items():
            fig.add_hline(y=lvl, line_dash="dash", opacity=0.3, annotation_text=name)

        buys = df[df["signal"] == "BUY"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], mode="markers",
                                     name="BUY", marker_symbol="triangle-up", marker_size=10))
        # live marker
        now_utc = pd.Timestamp.now(tz="UTC")
        fig.add_trace(go.Scatter(
            x=[now_utc], y=[live_price_val],
            mode="markers+text", text=[f"{live_price_val:.2f}"],
            textposition="top center", name="Live", marker_symbol="circle", marker_size=10
        ))
        fig.add_vline(x=bar_close, line_dash="dot", opacity=0.3)

        fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
        chart_ph.plotly_chart(fig, use_container_width=True)

    # --- MACD & Prob charts (κάτω, σταθερά) ---
    cL, cR = st.columns(2)
    if {"macd","macd_signal"}.issubset(df.columns):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal"))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        cL.plotly_chart(fig2, use_container_width=True)

    if "prob_up" in df.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df["prob_up"], name="Prob(up)"))
        fig3.add_hline(y=thr, line_dash="dash", opacity=0.4, annotation_text="BUY thr")
        fig3.add_hline(y=1-thr, line_dash="dash", opacity=0.4, annotation_text="SELL thr")
        fig3.update_yaxes(range=[0, 1])
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        cR.plotly_chart(fig3, use_container_width=True)

    # --- Last decision ---
    last_sig = (df["signal"].iloc[-1] or df.get("signal_prob", "").iloc[-1] or "")
    st.info(f"Last decision: **{last_sig or 'No signal'}**  |  time: `{ts_open}`")

    # ===== Live update loop (1s) =====
    # Τρέχει για poll φορές, χωρίς να εξαφανίζονται τα charts.
    for _ in range(int(poll)):
        live_price = lt.latest_price or float(last["close"])
        price_ph.metric("Price (live)", f"{live_price:.2f}")
        render_chart(live_price)
        time.sleep(1)

    # Μετά το μικρό loop → πλήρες rerun (αν χρειαστεί, θα ξανατραβήξει δεδομένα)
    st.rerun()

except Exception as e:
    st.error(f"Error: {e}")
