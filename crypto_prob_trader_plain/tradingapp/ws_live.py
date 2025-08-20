# ws_live.py
import threading
from binance import ThreadedWebsocketManager
from config import settings

class LiveTicker:
    """
    Binance trade WebSocket: κρατάει την τελευταία τιμή σε πραγματικό χρόνο.
    Public stream (δεν χρειάζονται API keys για ανάγνωση).
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.latest_price: float | None = None
        self.latest_ts: int | None = None
        self._lock = threading.Lock()
        self._twm: ThreadedWebsocketManager | None = None
        self._started = False

    def _on_msg(self, msg: dict):
        # trade event → price 'p' (string), trade time 'T'
        if msg.get("e") == "trade":
            try:
                p = float(msg["p"])
                t = int(msg["T"])
            except Exception:
                return
            with self._lock:
                self.latest_price = p
                self.latest_ts = t

    def start(self):
        if self._started:
            return
        # testnet flag από settings (αν το spot testnet δεν εκπέμπει, βάλε BINANCE_TESTNET=false)
        self._twm = ThreadedWebsocketManager(testnet=settings.testnet)
        self._twm.start()
        self._twm.start_trade_socket(callback=self._on_msg, symbol=self.symbol)
        self._started = True

    def stop(self):
        if self._twm:
            self._twm.stop()
            self._twm = None
        self._started = False
