
# Crypto Prob Trader (Plain Files)

Project με Python scripts για live chart/trading και backtests σε Binance,
συνδυάζοντας **κανόνες** (RSI/EMA/MACD/Fibonacci) και **πιθανοτικό μοντέλο** (λογιστική παλινδρόμηση).

> ⚠️ Εκπαιδευτική χρήση. Default: PAPER_TRADING=true — δε στέλνει πραγματικές εντολές.

## Εγκατάσταση
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Τρέξιμο
```bash
python backtest.py     # σύγκριση rule / prob / hybrid
python run_chart.py    # live γράφημα με P(up)
python run_live.py     # live loop (hybrid απόφαση)
```

## Ρυθμίσεις (.env)
- SYMBOL, INTERVAL, LIMIT
- RSI_LEN, FIB_LOOKBACK, PROX_PCT
- PROB_THRESHOLD (π.χ. 0.55)
- PAPER_TRADING=true
- BINANCE_TESTNET=true
- ORDER_SIZE_USDT=50
- BINANCE_API_KEY, BINANCE_API_SECRET
- POLL_SECONDS=60

## Futures
Για leverage χρειάζεται ξεχωριστός client (UMFutures) και διαχείριση θέσεων.
Προς το παρόν γίνεται spot/paper για ασφάλεια.
