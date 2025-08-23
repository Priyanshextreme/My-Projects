# Stock Market Tracker (Streamlit)
# Run: streamlit run app.py

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------- Small helper: page auto-refresh WITHOUT extra packages ----------
def auto_refresh(seconds: int):
    # Refresh the whole page every N seconds (simple + reliable)
    st.markdown(f"<meta http-equiv='refresh' content='{int(seconds)}'>", unsafe_allow_html=True)

# ---------- Indicators ----------
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

# ---------- Page ----------
st.set_page_config(page_title="Stock Market Tracker", page_icon="📈", layout="wide")
st.title("📈 Stock Market Tracker")
st.caption("Examples: US `AAPL, MSFT, TSLA` • India add `.NS` like `RELIANCE.NS, TCS.NS` • Cotton futures: `CT=F`")

with st.sidebar:
    st.header("⚙️ Controls")

    # Ticker input that remembers last value across refresh
    tickers_text = st.text_input(
        "Ticker(s), comma-separated",
        value=st.session_state.get("last_tickers", "AAPL, MSFT, TSLA")
    )
    st.session_state["last_tickers"] = tickers_text

    today = dt.date.today()
    start_date = st.date_input("Start date", value=today - dt.timedelta(days=365), max_value=today)
    end_date = st.date_input("End date", value=today, max_value=today)

    # All valid yfinance intervals (minute + daily/weekly/monthly)
    interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"]
    interval = st.selectbox("Interval", interval_options, index=0)
    st.caption("Note: 1m data ≈ last 7 days; other minute bars have limited history vs daily.")

    indicators = st.multiselect(
        "Indicators", ["SMA 20", "SMA 50", "EMA 20", "EMA 50", "RSI 14"],
        default=["SMA 20", "SMA 50"]
    )
    show_volume = st.checkbox("Show volume", True)

    st.divider()
    live_mode = st.checkbox("🔴 Live mode (auto-refresh, 1-minute bars)", value=False,
                            help="Uses 1-minute data for today and refreshes the page.")
    refresh_sec = st.number_input("Refresh every (seconds)", min_value=5, max_value=120, value=15, step=5)
    if live_mode:
        auto_refresh(int(refresh_sec))

# ---------- Data helpers ----------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end + dt.timedelta(days=1),
        interval=interval, progress=False
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns like ('Close','RELIANCE.NS') -> 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[cols]

def fetch_intraday_1m(ticker: str) -> pd.DataFrame:
    """Fetch 1-minute bars for today (subject to Yahoo delays)."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols]
    except Exception:
        return pd.DataFrame()

def clamp_intraday_start(start_date: dt.date, interval: str) -> dt.date:
    """Yahoo limits intraday history. Keep requests in-range to avoid empty data."""
    intraday_set = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    if interval not in intraday_set:
        return start_date
    today = dt.date.today()
    if interval == "1m":
        limit_days = 7          # ~7 days of 1-minute data
    else:
        limit_days = 60         # safe default for other minute bars
    min_start = today - dt.timedelta(days=limit_days)
    return max(start_date, min_start)

# ---------- Chart helpers ----------
def plot_price(df: pd.DataFrame, label: str, show_volume_flag: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name=f"{label} Close"))
    if show_volume_flag and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name=f"{label} Volume", opacity=0.3, yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def add_indicator_traces(fig: go.Figure, df: pd.DataFrame, indicators_list: list, label: str):
    close = df["Close"]
    for ind in indicators_list:
        if ind.startswith("SMA"):
            w = int(ind.split()[1])
            fig.add_trace(go.Scatter(x=df["Date"], y=sma(close, w), mode="lines", name=f"{label} {ind}"))
        elif ind.startswith("EMA"):
            w = int(ind.split()[1])
            fig.add_trace(go.Scatter(x=df["Date"], y=ema(close, w), mode="lines", name=f"{label} {ind}"))

def rsi_chart(df: pd.DataFrame, label: str):
    r = rsi(df["Close"], 14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=r, mode="lines", name=f"{label} RSI(14)"))
    fig.add_hline(y=70, line_dash="dot")
    fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(title="RSI"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Main ----------
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
if not tickers:
    st.stop()

for t in tickers:
    st.subheader(f"📊 {t}")

    # Choose data source
    if live_mode:
        df = fetch_intraday_1m(t)
        effective_interval = "1m"
        effective_start = dt.date.today()
    else:
        effective_interval = interval
        adj_start = clamp_intraday_start(start_date, interval)
        df = fetch_history(t, adj_start, end_date, interval)
        effective_start = adj_start

    if df.empty:
        st.warning("No data fetched. Check ticker, interval, or date range.")
        continue  # keep looping other tickers

    # Top metrics (last price + change)
    last = float(df.iloc[-1]["Close"])
    prev = float(df.iloc[-2]["Close"]) if len(df) > 1 else last
    st.metric("Last Price", f"{last:,.2f}", f"{((last - prev) / prev) * 100:+.2f}%")

    # Price plot + indicators
    fig = plot_price(df, t, show_volume)
    add_indicator_traces(fig, df, [i for i in indicators if "RSI" not in i], t)
    st.plotly_chart(fig, use_container_width=True)

    # RSI panel
    if "RSI 14" in indicators:
        rsi_chart(df, t)

    # Little badge showing what interval/date range was effectively used
    st.caption(f"Interval: **{effective_interval}** • Start used: **{effective_start}** • End: **{end_date if not live_mode else dt.date.today()}**")

    with st.expander("Show raw data & download"):
        st.dataframe(df.tail(200))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {t} CSV",
            data=csv,
            file_name=f"{t}_{effective_start}_{end_date if not live_mode else dt.date.today()}_{effective_interval}.csv",
            mime="text/csv"
        )
