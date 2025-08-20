import time
from math import isclose
import pandas as pd

from config import settings
from datafeed import get_klines
from strategy import generate_signals
from execute import place_order
from features import make_features
from model import fit_prob_model, add_probabilities

MAX_SL_PCT = 0.10
MAX_TP_PCT = 0.20
LEVERAGE   = 13

last_signal_time = None
prev_lvls = None
_model = None
_feats = None

def lvls_changed(old, new, tol=0.01):
    if old is None:
        return True
    if set(old.keys()) != set(new.keys()):
        return True
    for k in new.keys():
        if not isclose(float(new[k]), float(old[k]), abs_tol=tol):
            return True
    return False

def nearest_fib(price: float, lvls: dict):
    best_name, best_val, best_pct = None, None, None
    for name, lvl in lvls.items():
        pct = abs(price - lvl) / lvl * 100.0
        if best_pct is None or pct < best_pct:
            best_name, best_val, best_pct = name, lvl, pct
    return best_name, best_val, best_pct

def ensure_probabilities(df: pd.DataFrame):
    global _model, _feats
    feat_df, feats = make_features(
        df, rsi_len=settings.rsi_len,
        lookback=settings.fib_lookback,
        prox_pct=settings.prox_pct
    )
    if _model is None or _feats != feats:
        _model = fit_prob_model(feat_df, feats)
        _feats = feats
    out = add_probabilities(_model, feat_df, feats, thr=settings.threshold)
    return out

def interval_to_timedelta(interval_str: str) -> pd.Timedelta:
    """Μετατρέπει '1m','5m','1h','4h','1d' σε Timedelta."""
    unit = interval_str[-1].lower()
    n = int(interval_str[:-1])
    if unit == 'm':
        return pd.Timedelta(minutes=n)
    if unit == 'h':
        return pd.Timedelta(hours=n)
    if unit == 'd':
        return pd.Timedelta(days=n)
    raise ValueError(f"Unsupported interval {interval_str}")

def main():
    global last_signal_time, prev_lvls
    print(f"Running live (poll={settings.poll_seconds}s) on {settings.symbol} {settings.interval}")
    while True:
        try:
            # df.index = open_time (UTC) και έχουμε στήλη close_time από το datafeed
            df = get_klines(settings.symbol, settings.interval, settings.limit)
            df, lvls = generate_signals(df, settings.rsi_len, settings.fib_lookback, settings.prox_pct)

            # Προβλέψεις πιθανοτήτων
            prob_df = ensure_probabilities(df)
            p_up = float(prob_df["prob_up"].iloc[-1])

            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            ts_open  = df.index[-1]                   # open time του κεριού (UTC)
            price    = float(last_row["close"])
            rsi      = float(last_row.get("rsi", float("nan")))
            ema50    = float(last_row.get("ema50", float("nan")))
            ema200   = float(last_row.get("ema200", float("nan")))
            atr      = float(last_row.get("atr", float("nan")))
            macd     = float(last_row.get("macd", float("nan")))
            macd_sig = float(last_row.get("macd_signal", float("nan")))

            # ---- Ώρα τώρα & αντίστροφη μέτρηση μέχρι κλείσιμο κεριού ----
            now_utc = pd.Timestamp.now(tz="UTC")

            if "close_time" in df.columns:
                bar_close = last_row["close_time"]
            else:
                bar_close = ts_open + interval_to_timedelta(settings.interval)

            # Βεβαιώσου ότι το bar_close είναι tz-aware UTC
            if not isinstance(bar_close, pd.Timestamp):
                bar_close = pd.to_datetime(bar_close, utc=True)
            elif bar_close.tzinfo is None:
                bar_close = bar_close.tz_localize("UTC")

            eta = max(pd.Timedelta(0), bar_close - now_utc)
            print(f"[now={now_utc:%Y-%m-%d %H:%M:%S}Z] bar_open={ts_open} | bar_close={bar_close} | ETA={eta}")

            # ---- Εμφάνιση επιπέδων/δεικτών ----
            if lvls_changed(prev_lvls, lvls):
                print("\n--- Fibonacci Levels ---")
                for level, lvl_price in lvls.items():
                    print(f"{level}: {lvl_price:.2f}")
                prev_lvls = lvls

            print(f"RSI: {rsi:.2f} | EMA50: {ema50:.2f} | EMA200: {ema200:.2f} | ATR: {atr:.2f}")
            print(f"MACD: {macd:.2f} | Signal: {macd_sig:.2f} | Price: {price:.2f} | Prob(up): {p_up:.2f}")

            near_name, near_val, near_pct = nearest_fib(price, lvls)
            print(f"Nearest Fib: {near_name} @ {near_val:.2f} (dist {near_pct:.2f}%)")

            # ---- Κανόνες ----
            golden_cross = (last_row["ema50"] > last_row["ema200"]) and (prev_row["ema50"] <= prev_row["ema200"])
            rsi_ok  = rsi < 60
            prox_ok = near_pct <= float(settings.prox_pct)
            macd_ok = macd > macd_sig

            all_ok_buy = golden_cross and rsi_ok and prox_ok and macd_ok
            decision = all_ok_buy and (p_up > settings.threshold)

            # Χρησιμοποιούμε το open timestamp για να μην ξαναπάρουμε διπλό σήμα στο ίδιο κερί
            if decision and last_signal_time != ts_open:
                recent_low = float(df["low"].tail(20).min())

                sl_candidates = []
                if atr == atr:
                    sl_candidates.append(price - 1.5 * atr)
                sl_candidates.append(recent_low)
                sl_candidates.append(near_val * (1 - float(settings.prox_pct) / 100.0))

                max_sl_price = price * (1 - MAX_SL_PCT)
                stop_loss = max(max_sl_price, min(sl_candidates))

                higher_fibs = [v for v in lvls.values() if v > price]
                if higher_fibs:
                    tp1 = min(higher_fibs)
                elif atr == atr:
                    tp1 = price + 1.5 * atr
                else:
                    tp1 = price * 1.01
                max_tp_price = price * (1 + MAX_TP_PCT)
                tp1 = min(tp1, max_tp_price)

                risk = max(1e-6, price - stop_loss)
                tp2 = min(price + 2.0 * risk, max_tp_price)

                sl_pct  = (price - stop_loss) / price * 100.0
                tp1_pct = (tp1 / price - 1) * 100.0
                tp2_pct = (tp2 / price - 1) * 100.0

                sl_lev  = sl_pct  * LEVERAGE
                tp1_lev = tp1_pct * LEVERAGE
                tp2_lev = tp2_pct * LEVERAGE

                print(f"[{ts_open}] ✅ BUY @ {price:.2f}  |  x{LEVERAGE} lev | P(up)={p_up:.2f}")
                print(f"   SL:  {stop_loss:.2f}  ({sl_pct:.2f}%  | lev≈{sl_lev:.1f}%)")
                print(f"   TP1: {tp1:.2f}  ({tp1_pct:.2f}% | lev≈{tp1_lev:.1f}%)")
                print(f"   TP2: {tp2:.2f}  ({tp2_pct:.2f}% | lev≈{tp2_lev:.1f}%)")

                place_order("BUY", settings.symbol, settings.order_size_usdt)
                last_signal_time = ts_open
            else:
                reasons = []
                if not all_ok_buy: reasons.append("rule-fail")
                if not (p_up > settings.threshold): reasons.append("prob<thr")
                print(f"[{ts_open}] No signal ({', '.join(reasons)})")

        except Exception as e:
            print("Error:", e)

        time.sleep(settings.poll_seconds)

if __name__ == "__main__":
    main()
