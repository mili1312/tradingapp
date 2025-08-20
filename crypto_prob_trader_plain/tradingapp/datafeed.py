
import pandas as pd
from binance.client import Client
from config import settings

INTERVAL_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "3m": Client.KLINE_INTERVAL_3MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "2h": Client.KLINE_INTERVAL_2HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "6h": Client.KLINE_INTERVAL_6HOUR,
    "8h": Client.KLINE_INTERVAL_8HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}

def get_client():
    if settings.testnet:
        client = Client(settings.api_key, settings.api_secret, testnet=True)
    else:
        client = Client(settings.api_key, settings.api_secret)
    return client

def get_klines(symbol: str = None, interval: str = None, limit: int = None) -> pd.DataFrame:
    symbol = symbol or settings.symbol
    interval = interval or settings.interval
    limit = limit or settings.limit
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval {interval}")
    client = get_client()
    raw = client.get_klines(symbol=symbol, interval=INTERVAL_MAP[interval], limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    # Convert types
    for c in ["open","high","low","close","volume","qav","tbbav","tbqav"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    df.rename_axis("time", inplace=True)
    return df[["open","high","low","close","volume"]].copy()
