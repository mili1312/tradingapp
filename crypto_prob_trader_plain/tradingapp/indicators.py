
import pandas as pd
import pandas_ta as ta

def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = ta.rsi(out["close"], length=length)
    return out

def add_ema(df: pd.DataFrame, spans=(50, 200)) -> pd.DataFrame:
    out = df.copy()
    for s in spans:
        out[f"ema{s}"] = out["close"].ewm(span=s, adjust=False).mean()
    return out

def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    out["atr"] = true_range.rolling(length, min_periods=length).mean()
    return out

def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out

def fib_levels(df: pd.DataFrame, lookback: int = 200) -> dict:
    window = df.tail(lookback)
    hi = float(window["high"].max())
    lo = float(window["low"].min())
    rng = hi - lo
    return {
        "fib382": hi - 0.382 * rng,
        "fib50":  hi - 0.5   * rng,
        "fib618": hi - 0.618 * rng,
    }

def near_any_fib(price: float, levels: dict, prox_pct: float = 0.25) -> bool:
    for lvl in levels.values():
        if abs(price - lvl) / lvl * 100.0 <= prox_pct:
            return True
    return False
