from ib_insync import IB, Stock, util
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

ib = IB()

print("Connecting to IBKR...")
ib.connect(HOST, PORT, clientId=CLIENT_ID)

contract = Stock('AAPL', 'SMART', 'USD')

print("Fetching AAPL historical data...")

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='3 Y',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True
)

# convert to dataframe
df = util.df(bars)

print(df.head())

# create folder
os.makedirs("data/market_data", exist_ok=True)

file_path = "data/market_data/AAPL_1D.parquet"

df.to_parquet(file_path)

print(f"Saved to {file_path}")

ib.disconnect()
print("Done")
