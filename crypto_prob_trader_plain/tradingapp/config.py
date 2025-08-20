
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    # Market / data
    symbol: str = os.getenv("SYMBOL", "ETHUSUSDT".replace("USUSDT","USDT"))  # safety for typo
    interval: str = os.getenv("INTERVAL", "1h")
    limit: int = int(os.getenv("LIMIT", "1000"))

    # Indicators / features
    rsi_len: int = int(os.getenv("RSI_LEN", "14"))
    fib_lookback: int = int(os.getenv("FIB_LOOKBACK", "200"))
    prox_pct: float = float(os.getenv("PROX_PCT", "0.25"))  # % distance to fib

    # Probabilities
    threshold: float = float(os.getenv("PROB_THRESHOLD", "0.55"))

    # Execution & env
    paper: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    testnet: bool = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    order_size_usdt: float = float(os.getenv("ORDER_SIZE_USDT", "50"))

    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")

    # polling
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))

settings = Settings()
