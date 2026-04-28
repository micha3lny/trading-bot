from ib_insync import IB
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

ib = IB()

print("Connecting to IBKR...")
ib.connect(HOST, PORT, clientId=CLIENT_ID)

print("Connected:", ib.isConnected())

# Fetch server time
print("Server time:", ib.reqCurrentTime())

# Fetch accounts
accounts = ib.managedAccounts()
print("Accounts:", accounts)

ib.disconnect()
print("Disconnected")
