
from config import settings
from datafeed import get_client

def place_order(side: str, symbol: str, quote_usdt: float):
    """Place a market order.
    PAPER_TRADING=true: just print & return a stub.
    Else: attempt spot market order (quote order qty).
    NOTE: For leverage/futures, integrate UMFutures separately.
    """
    side = side.upper()
    if settings.paper:
        print(f"[PAPER] {side} {symbol} for ~{quote_usdt} USDT (no real order sent)")
        return {"paper": True, "side": side, "symbol": symbol, "quote": quote_usdt}

    client = get_client()
    try:
        if side == "BUY":
            res = client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quoteOrderQty=str(quote_usdt),
            )
        elif side == "SELL":
            raise NotImplementedError("SELL requires handling position/base qty.")
        else:
            raise ValueError("side must be BUY or SELL")
        print("Order response:", res)
        return res
    except Exception as e:
        print("Order error:", e)
        return {"error": str(e)}
