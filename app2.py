# Stock Market Tracker (Streamlit)
# Run: streamlit run app.py

import datetime as dt
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------------------
# Plotly config
# -------------------------------
PLOT_CONFIG = {
    "scrollZoom": True,      # mouse wheel zoom
    "doubleClick": "reset",  # double-click resets
    "displaylogo": False,    # hide plotly logo
    "responsive": True,
}

def enhance_interactivity(fig: go.Figure, add_rangeslider: bool = True, default_drag: str = "pan") -> go.Figure:
    fig.update_layout(
        dragmode=default_drag,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if add_rangeslider:
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=5, label="5D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid")
    fig.update_yaxes(showspikes=True, spikemode="across", spikedash="solid")
    return fig

# -------------------------------
# Utils
# -------------------------------
def parse_tickers(text: str) -> list[str]:
    parts = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

@st.cache_data(ttl=600, show_spinner=False)
def yahoo_symbol_search(query: str) -> pd.DataFrame:
    if not query or len(query) < 2:
        return pd.DataFrame()
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 25, "newsCount": 0, "quotesQueryId": "tss_match_query"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("quotes", [])
        rows = []
        for q in data:
            rows.append({
                "Symbol": q.get("symbol"),
                "Name": q.get("shortname") or q.get("longname"),
                "Type": q.get("quoteType"),
                "Exchange": q.get("exchDisp"),
                "Region": q.get("region"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# -------------------------------
# Indicators
# -------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df[["Date", "Open", "High", "Low", "Close"]].copy()
    ha["HA_Close"] = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4
    ha["HA_Open"] = 0.0
    if len(ha) > 0:
        ha.loc[0, "HA_Open"] = (ha.loc[0, "Open"] + ha.loc[0, "Close"]) / 2
        for i in range(1, len(ha)):
            ha.loc[i, "HA_Open"] = (ha.loc[i-1, "HA_Open"] + ha.loc[i-1, "HA_Close"]) / 2
    ha["HA_High"] = ha[["HA_Open", "HA_Close", "High"]].max(axis=1)
    ha["HA_Low"]  = ha[["HA_Open", "HA_Close", "Low"]].min(axis=1)
    return ha

# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Stock Market Tracker", page_icon="📈", layout="wide")
st.title("📈 Stock Market Tracker")
st.caption("Examples: US `AAPL, MSFT, TSLA` • India add `.NS` like `RELIANCE.NS` • Futures: `CT=F`")

with st.sidebar:
    st.header("⚙️ Controls")

    st.markdown("**Search by name**")
    q = st.text_input("Company / crypto / fund name", placeholder="e.g., Reliance, Apple")
    res = yahoo_symbol_search(q)
    if not res.empty:
        choices = [f"{row.Symbol} — {row.Name or '—'} ({row.Exchange})" for _, row in res.iterrows()]
        sel = st.selectbox("Matches", choices, index=0)
        chosen_symbol = sel.split(" — ", 1)[0].strip() if sel else None
        if st.button("➕ Add symbol", use_container_width=True) and chosen_symbol:
            current = parse_tickers(st.session_state.get("last_tickers", "AAPL, MSFT, TSLA"))
            if chosen_symbol not in current:
                current.append(chosen_symbol)
                st.session_state["last_tickers"] = ", ".join(current)
                st.success(f"Added {chosen_symbol}")
            else:
                st.info(f"{chosen_symbol} already in list")

    st.divider()

    tickers_text = st.text_input(
        "Ticker(s), comma-separated",
        value=st.session_state.get("last_tickers", "AAPL, MSFT, TSLA"),
        help="Tip: NSE India needs `.NS` suffix"
    )
    st.session_state["last_tickers"] = tickers_text

    today = dt.date.today()
    start_date = st.date_input("Start date", value=today - dt.timedelta(days=365))
    end_date = st.date_input("End date", value=today)

    interval_options = ["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"]
    interval = st.selectbox("Interval", interval_options, index=0)

    st.subheader("Chart style")
    chart_style = st.selectbox("Type", ["Line", "Area", "Candlestick", "OHLC", "Heikin-Ashi"], index=2)

    indicators = st.multiselect(
        "Indicators", ["SMA 20", "SMA 50", "EMA 20", "EMA 50", "RSI 14"], default=["SMA 20", "SMA 50"]
    )
    show_volume = st.checkbox("Show volume (Line/Area)", True)

    st.divider()
    live_mode = st.checkbox("🔴 Live mode (auto-refresh 1m)", value=False)
    refresh_sec = st.number_input("Refresh every (seconds)", 5, 120, 15, 5)

# -------------------------------
# Data helpers
# -------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_history(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    try:
        if interval.endswith("m"):  # intraday
            df = yf.download(ticker, period="7d", interval=interval, progress=False, threads=False)
        else:
            df = yf.download(ticker, start=start, end=end + dt.timedelta(days=1),
                             interval=interval, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df

def fetch_intraday_1m(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df.empty: return pd.DataFrame()
        df = df.reset_index().rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------
# Charting
# -------------------------------
def plot_price(df: pd.DataFrame, label: str, style: str, show_volume_flag: bool):
    fig = go.Figure()
    if style == "Heikin-Ashi":
        ha = heikin_ashi(df)
        fig.add_trace(go.Candlestick(x=ha["Date"], open=ha["HA_Open"], high=ha["HA_High"],
                                     low=ha["HA_Low"], close=ha["HA_Close"],
                                     name=f"{label} Heikin-Ashi"))
        plot_close = ha["HA_Close"]
    elif style == "Candlestick":
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                     low=df["Low"], close=df["Close"], name=f"{label} Candles"))
        plot_close = df["Close"]
    elif style == "OHLC":
        fig.add_trace(go.Ohlc(x=df["Date"], open=df["Open"], high=df["High"],
                              low=df["Low"], close=df["Close"], name=f"{label} OHLC"))
        plot_close = df["Close"]
    elif style == "Area":
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", fill="tozeroy", name=f"{label} Close"))
        plot_close = df["Close"]
    else:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name=f"{label} Close"))
        plot_close = df["Close"]

    if show_volume_flag and "Volume" in df.columns and style in {"Line", "Area"}:
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", opacity=0.3, yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False))

    return fig, plot_close

# -------------------------------
# Main
# -------------------------------
tickers = parse_tickers(st.session_state.get("last_tickers", tickers_text))
if not tickers: st.stop()

for t in tickers:
    st.subheader(f"📊 {t}")
    df = fetch_intraday_1m(t) if live_mode else fetch_history(t, start_date, end_date, interval)
    if df.empty:
        st.warning("No data found.")
        continue

    last, prev = float(df.iloc[-1]["Close"]), float(df.iloc[-2]["Close"])
    st.metric("Last Price", f"{last:,.2f}", f"{((last-prev)/prev)*100:+.2f}%")

    fig, plot_close = plot_price(df, t, chart_style, show_volume)
    for ind in [i for i in indicators if "RSI" not in i]:
        if ind.startswith("SMA"):
            w = int(ind.split()[1]); fig.add_trace(go.Scatter(x=df["Date"], y=sma(plot_close, w), mode="lines", name=ind))
        elif ind.startswith("EMA"):
            w = int(ind.split()[1]); fig.add_trace(go.Scatter(x=df["Date"], y=ema(plot_close, w), mode="lines", name=ind))

    fig = enhance_interactivity(fig)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    if "RSI 14" in indicators and len(df) >= 14:
        r = rsi(df["Close"], 14)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df["Date"], y=r, mode="lines", name="RSI(14)"))
        rsi_fig.add_hline(y=70, line_dash="dot"); rsi_fig.add_hline(y=30, line_dash="dot")
        rsi_fig = enhance_interactivity(rsi_fig, add_rangeslider=False)
        st.plotly_chart(rsi_fig, use_container_width=True, config=PLOT_CONFIG)

# -------------------------------
# Live refresh countdown
# -------------------------------
if live_mode:
    placeholder = st.empty()
    for remaining in range(int(refresh_sec), 0, -1):
        placeholder.caption(f"🔴 Live mode — updating in {remaining} sec…")
        time.sleep(1)
    try: st.rerun()
    except Exception: st.experimental_rerun()
