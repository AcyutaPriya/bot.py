import os
import sys
import time
import requests
import traceback
from datetime import datetime, timedelta, date
from typing import Literal, Tuple, List, Dict
import pandas as pd
from pydantic import BaseModel, Field

# Ensure output prints immediately in GitHub Actions logs
sys.stdout.reconfigure(line_buffering=True)

# --- Alpaca SDK ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.common.exceptions import APIError

# --- OpenAI SDK ---
from openai import OpenAI

# ==============================================================================
# 1. CREDENTIALS & INITIALIZATION
# ==============================================================================
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "").strip()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing Alpaca API credentials in GitHub Secrets.")

if not OPENAI_KEY:
    raise ValueError("Missing OPENAI_API_KEY in GitHub Secrets.")

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
ai_client = OpenAI(api_key=OPENAI_KEY)


# ==============================================================================
# 2. FINNHUB MACROECONOMIC BLACKOUT ENGINE
# ==============================================================================
class FinnhubCalendarFilter:
    BASE_URL = "https://finnhub.io/api/v1/calendar/economic"
    CRITICAL_KEYWORDS = [
        "cpi", "consumer price index", "fomc", "fed interest rate",
        "nonfarm payrolls", "non-farm payrolls", "gdp", "pce price index",
        "unemployment rate", "powell speaks"
    ]

    @classmethod
    def is_macro_blackout_today(cls) -> Tuple[bool, str]:
        if not FINNHUB_KEY:
            print("[i] No FINNHUB_API_KEY provided. Skipping news check.")
            return False, ""

        today_str = date.today().isoformat()
        try:
            res = requests.get(
                cls.BASE_URL,
                params={"from": today_str, "to": today_str, "token": FINNHUB_KEY
