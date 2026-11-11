import os
import sys
import requests
import traceback
from datetime import datetime, timedelta, date
from typing import Literal, Tuple, List, Dict
import pandas as pd
from pydantic import BaseModel, Field

# Force unbuffered output so print statements display instantly in GitHub Actions logs
sys.stdout.reconfigure(line_buffering=True)

# --- Alpaca SDK ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.common.exceptions import APIError

# --- AI SDK ---
from google import genai
from google.genai import types

# ==============================================================================
# 1. CREDENTIALS & INITIALIZATION
# ==============================================================================
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "").strip()
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing Alpaca API credentials in GitHub Secrets.")

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)


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
            print("[i] No FINNHUB_API_KEY provided. Skipping economic news calendar check.")
            return False, ""

        today_str = date.today().isoformat()
        try:
            res = requests.get(
                cls.BASE_URL, 
                params={"from": today_str, "to": today_str, "token": FINNHUB_KEY}, 
                timeout=10
            )
            res.raise_for_status()
            events = res.json().get("economicCalendar", [])
        except Exception as e:
            print(f"[!] Warning: Finnhub API check skipped ({e}). Defaulting to open trading.")
            return False, ""

        for item in events:
            if item.get("country", "").upper() != "US":
                continue
            event_name = item.get("event", "")
            impact = str(item.get("impact", "")).lower()
            
            is_high_impact = impact in ["high", "3"]
            keyword_match = any(kw in event_name.lower() for kw in cls.CRITICAL_KEYWORDS)

            if is_high_impact or keyword_match:
                return True, f"{event_name} (Impact: {impact})"

        return False, ""


# ==============================================================================
# 3. POSITION RISK POLICY DURING BLACKOUTS
# ==============================================================================
def handle_blackout_positions(symbol: str, blackout_policy: str = "TIGHTEN_OR_FLATTEN"):
    try:
        position = trading_client.get_open_position(symbol)
    except APIError:
        print(f"[*] No open position in {symbol} to manage.")
        return

    qty = int(position.qty)
    entry_price = float(position.avg_entry_price)
    current_price = float(position.current_price)
    unrealized_pl = float(position.unrealized_pl)

    print(f"[*] Active Position: {qty} shares of {symbol} | Unrealized P&L: ${unrealized_pl:+.2f}")

    if blackout_policy == "FLATTEN":
        print(f"[!] FLATTEN: Closing all {symbol} positions to stay 100% in cash.")
        trading_client.close_position(symbol)

    elif blackout_policy == "TIGHTEN_OR_FLATTEN":
        if current_price > entry_price * 1.005:
            print(f"[+] Position is in profit. Moving Stop Loss to Breakeven (${entry_price:.2f})...")
            trading_client.cancel_orders()
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_loss=StopLossRequest(stop_price=round(entry_price, 2))
                )
            )
        else:
            print(f"[-] Position is flat or in loss. Liquidating before news volatility...")
            trading_client.close_position(symbol)


# ==============================================================================
# 4. DATA EXTRACTION (COMPATIBLE WITH FREE ALPACA TIERS)
# ==============================================================================
def fetch_bars_safe(symbol: str, timeframe: TimeFrame, days_back: int) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        feed=DataFeed.IEX
    )
    bars = data_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars


def extract_mtf_icc_metrics(symbol: str) -> dict:
    try:
        # Pull 1-hour bars for HTF structure
