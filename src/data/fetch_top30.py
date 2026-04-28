from ib_insync import IB, Stock, util
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

TICKERS = [
    "AAPL","MSFT","NVDA","AMD","META","GOOGL","AMZN","TSLA","NFLX","INTC",
    "AVGO","ADBE","CRM","CSCO","QCOM","TXN","MU","AMAT","PYPL","SHOP",
    "PLTR","SNOW","UBER","LYFT","COIN","SQ","ROKU","ZM","DOCU","PINS"
]

ib = IB()
print("Connecting to IBKR...")
ib.connect(HOST, PORT, clientId=CLIENT_ID)

os.makedirs("data/market_data", exist_ok=True)

for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    contract = Stock(ticker, 'SMART', 'USD')

    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='3 Y',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True
    )

    df = util.df(bars)

    file_path = f"data/market_data/{ticker}_1D.parquet"
    df.to_parquet(file_path)

    print(f"Saved {ticker}")

ib.disconnect()
print("Done")
