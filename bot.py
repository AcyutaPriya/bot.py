import os
import requests
from datetime import datetime, timedelta, date
from typing import Literal, Tuple, List, Dict
import pandas as pd
from pydantic import BaseModel, Field

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
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

# --- AI SDK (Gemini by default; switchable to OpenAI) ---
from google import genai
from google.genai import types

# ==============================================================================
# 1. CREDENTIALS & INITIALIZATION
# ==============================================================================
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not ALPACA_KEY or not ALPACA_SECRET:
    raise ValueError("Missing Alpaca API credentials in environment secrets.")

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
            res = requests.get(cls.BASE_URL, params={"from": today_str, "to": today_str, "token": FINNHUB_KEY}, timeout=10)
            res.raise_for_status()
            events = res.json().get("economicCalendar", [])
        except Exception as e:
            print(f"[!] Warning: Finnhub API check failed ({e}). Defaulting to open trading.")
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
        if current_price > entry_price * 1.005:  # In profit by >= 0.5%
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
# 4. MULTI-TIMEFRAME DATA GATHERING (4H & 15M)
# ==============================================================================
def fetch_dataframe(symbol: str, timeframe: TimeFrame, days_back: int) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start_date, end=end_date)
    bars = data_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    return bars


def extract_mtf_icc_metrics(symbol: str) -> dict:
    timeframe_4h = TimeFrame(4, TimeFrameUnit.Hour)
    timeframe_15m = TimeFrame(15, TimeFrameUnit.Minute)

    df_4h = fetch_dataframe(symbol, timeframe=timeframe_4h, days_back=35)
    df_15m = fetch_dataframe(symbol, timeframe=timeframe_15m, days_back=5)

    return {
        "symbol": symbol,
        "current_price": float(df_15m["close"].iloc[-1]),
        "htf_4h": {
            "close": float(df_4h["close"].iloc[-1]),
            "swing_high": float(df_4h["high"].iloc[-20:].max()),
            "swing_low": float(df_4h["low"].iloc[-20:].min()),
            "trend_sma": round(float(df_4h["close"].rolling(20).mean().iloc[-1]), 2),
        },
        "ltf_15m": {
            "breakout_trigger": float(df_15m["high"].iloc[-12:-3].max()),
            "correction_low": float(df_15m["low"].iloc[-8:].min()),
            "correction_high": float(df_15m["high"].iloc[-8:].max()),
        },
    }


# ==============================================================================
# 5. STRUCTURED AI AUDITOR & DECISION ENGINE
# ==============================================================================
class MultiTimeframeICCDecision(BaseModel):
    htf_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="4H structural impulse (Indication)")
    ltf_phase: Literal["CORRECTION", "CONTINUATION", "CHOP"] = Field(description="15M current phase")
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Action command. BUY only if 4H is Bullish and 15M triggers Continuation.")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0")
    reasoning: str = Field(description="Summary of ICC setup logic")
    suggested_stop_loss: float = Field(description="Stop loss price level below 15M correction low")
    suggested_take_profit: float = Field(description="Take profit price level with min 1:2 R:R")


def get_ai_decision(market_data: dict) -> MultiTimeframeICCDecision:
    prompt = f"""
    You are an automated risk and execution engine executing the ICC (Indication, Correction, Continuation) trading model:
    
    1. Indication (4H chart):
       - If 4H Close > Trend Baseline SMA and printing higher highs, HTF Bias is BULLISH.
       - Otherwise BEARISH or NEUTRAL.
    2. Correction vs Continuation (15M chart):
       - If pulling back or consolidating: Phase = CORRECTION. Action = HOLD. Never enter in correction.
       - If 4H is BULLISH and 15M breaks above the correction high: Phase = CONTINUATION. Action = BUY.
       - If conflicting, Action = HOLD.
    3. Risk Parameters:
       - Stop-loss MUST be slightly below the 15M correction low.
       - Take-profit MUST deliver at least a 1:2 Risk-to-Reward ratio.

    MARKET DATA FOR {market_data['symbol']}:
    {market_data}
    """

    if GEMINI_KEY:
        ai_client = genai.Client(api_key=GEMINI_KEY)
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MultiTimeframeICCDecision,
                temperature=0.1,
            ),
        )
        return MultiTimeframeICCDecision.model_validate_json(response.text)

    elif OPENAI_KEY:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        res = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a quantitative trading controller enforcing ICC rules."},
                {"role": "user", "content": prompt}
            ],
            response_format=MultiTimeframeICCDecision
        )
        return res.choices[0].message.parsed
    else:
        raise ValueError("Neither GEMINI_API_KEY nor OPENAI_API_KEY was found in secrets.")


# ==============================================================================
# 6. SELF-AUDIT: PERFORMANCE & RETURN RATE CHECK
# ==============================================================================
def run_performance_self_audit():
    account = trading_client.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    cash = float(account.cash)

    day_return_pct = ((equity - last_equity) / last_equity) * 100 if last_equity > 0 else 0.0

    print("\n" + "=" * 55)
    print("               PORTFOLIO PERFORMANCE AUDIT             ")
    print("=" * 55)
    print(f"Current Equity Value : ${equity:,.2f}")
    print(f"Available Cash       : ${cash:,.2f}")
    print(f"Session Return Rate  : {day_return_pct:+.2f}%")
    print("=" * 55 + "\n")


# ==============================================================================
# 7. MAIN BOT LOOP
# ==============================================================================
def run_bot(symbol: str = "SPY", qty: int = 1):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ICC Run for {symbol}...")

    # Step 1: Self performance audit
    run_performance_self_audit()

    # Step 2: High-impact economic news filter
    is_blocked, reason = FinnhubCalendarFilter.is_macro_blackout_today()
    if is_blocked:
        print(f"[BLOCKED] Trading halted today due to high-impact macro news: {reason}")
        handle_blackout_positions(symbol, blackout_policy="TIGHTEN_OR_FLATTEN")
        return

    # Step 3: Check open positions
    open_qty = 0
    try:
        open_qty = int(trading_client.get_open_position(symbol).qty)
    except APIError:
        open_qty = 0

    # Step 4: Extract MTF technical data & run AI model
    print(f"[*] Extracting 4H Indication and 15M Continuation metrics for {symbol}...")
    metrics = extract_mtf_icc_metrics(symbol)

    print("[*] Passing market structure to AI reasoning agent...")
    decision = get_ai_decision(metrics)

    print(f"\n[AI Verdict] Action: {decision.action} | Confidence: {decision.confidence_score * 100:.1f}%")
    print(f"4H Bias: {decision.htf_bias} | 15M Phase: {decision.ltf_phase}")
    print(f"Reasoning: {decision.reasoning}\n")

    # Step 5: Execute bracket orders if criteria are met
    if decision.action == "BUY" and open_qty == 0 and decision.confidence_score >= 0.75:
        if decision.suggested_stop_loss >= metrics["current_price"]:
            print("[!] Invalid Stop Loss returned by AI model. Aborting trade.")
            return

        print(f"[+] Placing BUY bracket order for {qty} share(s) of {symbol}...")
        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(decision.suggested_take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(decision.suggested_stop_loss, 2)),
        )
        order = trading_client.submit_order(order_req)
        print(f"[✓] Bracket Order placed on Alpaca! Order ID: {order.id}")

    elif decision.action == "SELL" and open_qty > 0:
        print(f"[-] AI signaled exit. Liquidating {open_qty} shares of {symbol}...")
        trading_client.close_position(symbol)
    else:
        print(f"[*] No trade executed. Holding {open_qty} shares of {symbol}.")


if __name__ == "__main__":
    run_bot(symbol="SPY", qty=1)
