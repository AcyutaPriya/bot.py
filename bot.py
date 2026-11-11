import os
import sys
import time
import requests
from datetime import datetime, timedelta, date
from typing import Literal, Tuple, List, Dict
import pandas as pd
from pydantic import BaseModel, Field

# Unbuffered output ensures GitHub Actions logs display immediately
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
        url = f"https://finnhub.io/api/v1/calendar/economic?from={today_str}&to={today_str}&token={FINNHUB_KEY}"
        
        try:
            res = requests.get(url, timeout=10)
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
def handle_blackout_positions(symbol: str):
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

    if current_price > (entry_price * 1.005):
        print(f"[+] Moving Stop Loss to Breakeven (${entry_price:.2f})...")
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
        print("[-] Position is flat or in loss. Liquidating before news volatility...")
        trading_client.close_position(symbol)


# ==============================================================================
# 4. BAR EXTRACTION
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
    df_1h = fetch_bars_safe(symbol, TimeFrame.Hour, days_back=30)
    df_1m = fetch_bars_safe(symbol, TimeFrame.Minute, days_back=5)

    df_15m = df_1m.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    df_4h = df_1h.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

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
# 5. OPENAI STRUCTURED DECISION ENGINE (GPT-4O-MINI)
# ==============================================================================
class MultiTimeframeICCDecision(BaseModel):
    htf_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="4H structural impulse")
    ltf_phase: Literal["CORRECTION", "CONTINUATION", "CHOP"] = Field(description="15M current phase")
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Action command")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0")
    reasoning: str = Field(description="Summary of ICC setup logic")
    suggested_stop_loss: float = Field(description="Stop loss price level")
    suggested_take_profit: float = Field(description="Take profit price level")


def get_ai_decision(market_data: dict) -> MultiTimeframeICCDecision:
    prompt = f"""
    You are an automated risk and execution engine executing the ICC (Indication, Correction, Continuation) model:
    1. Indication (4H): Bullish if Close > Trend SMA and higher highs; else Bearish/Neutral.
    2. Correction vs Continuation (15M):
       - Pulling back/range: Phase = CORRECTION. Action = HOLD.
       - 4H Bullish and 15M closes above correction high: Phase = CONTINUATION. Action = BUY.
       - Otherwise Action = HOLD.
    3. Risk: Stop-loss slightly below 15M correction low. Take-profit at least 1:2 R:R.

    MARKET DATA:
    {market_data}
    """

    for attempt in range(1, 4):
        try:
            completion = ai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a quantitative trading controller enforcing ICC rules."},
                    {"role": "user", "content": prompt}
                ],
                response_format=MultiTimeframeICCDecision,
                temperature=0.1
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"[!] OpenAI attempt {attempt} failed ({e}).")
            if attempt < 3:
                time.sleep(3)
            else:
                print("[!] OpenAI unavailable. Safely defaulting to HOLD.")
                return MultiTimeframeICCDecision(
                    htf_bias="NEUTRAL",
                    ltf_phase="CHOP",
                    action="HOLD",
                    confidence_score=0.0,
                    reasoning="AI unavailable. Defaulting to HOLD to preserve capital.",
                    suggested_stop_loss=0.0,
                    suggested_take_profit=0.0
                )


# ==============================================================================
# 6. PORTFOLIO AUDIT
# ==============================================================================
def run_performance_self_audit():
    account = trading_client.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    cash = float(account.cash)

    day_return_pct = 0.0
    if last_equity > 0:
        day_return_pct = ((equity - last_equity) / last_equity) * 100

    print("=" * 50)
    print("PORTFOLIO AUDIT")
    print(f"Current Equity: ${equity:,.2f}")
    print(f"Available Cash: ${cash:,.2f}")
    print(f"Today's Return: {day_return_pct:+.2f}%")
    print("=" * 50)


# ==============================================================================
# 7. MAIN RUNNER
# ==============================================================================
def run_bot(symbol: str = "SPY", qty: int = 1):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ICC Run for {symbol}...")

    # Step A: Portfolio Audit
    run_performance_self_audit()

    # Step B: News Check
    is_blocked, reason = FinnhubCalendarFilter.is_macro_blackout_today()
    if is_blocked:
        print(f"[BLOCKED] Trading halted today due to macro news: {reason}")
        handle_blackout_positions(symbol)
        return

    # Step C: Existing Position
    open_qty = 0
    try:
        open_qty = int(trading_client.get_open_position(symbol).qty)
    except APIError:
        open_qty = 0

    # Step D: Market Metrics & AI Decision
    print(f"[*] Extracting 4H and 15M metrics for {symbol}...")
    metrics = extract_mtf_icc_metrics(symbol)

    print("[*] Requesting OpenAI (GPT-4o-mini) analysis...")
    decision = get_ai_decision(metrics)

    print(f"\n[AI Verdict] Action: {decision.action} | Confidence: {decision.confidence_score * 100:.1f}%")
    print(f"4H Bias: {decision.htf_bias} | 15M Phase: {decision.ltf_phase}")
    print(f"Reasoning: {decision.reasoning}\n")

    # Step E: Order Execution
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
        print(f"[*] No trade executed. Current position: {open_qty} shares of {symbol}.")


if __name__ == "__main__":
    run_bot(symbol="SPY", qty=1)
