# Stock Tracker — layout toggle (vertical or 2-col grid), cards, logos, gapless intraday
# Run: streamlit run app.py

import datetime as dt
import time
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# =============== Page / CSS ===============
st.set_page_config(page_title="Stock Tracker", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container {padding-top:.6rem; padding-bottom:1rem; max-width:1400px;}
  [data-testid="stSidebar"] .sidebar-title{
    position:sticky; top:0; z-index:1000;
    background:#0b1220; border:1px solid #263043; border-radius:12px;
    padding:12px 14px; margin-bottom:10px;
    color:#e5e7eb; letter-spacing:.5px;
    font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
    font-weight:900; font-size:22px;
  }
  /* Card around each stock */
  .stock-card{
    background:#0c111a;
    border:1px solid #273447;
    border-radius:16px;
    padding:18px 20px;
    margin:22px 0;
    box-shadow: 0 6px 18px rgba(0,0,0,.25);
  }
  .stock-card.compact { margin:14px 6px; }  /* used inside grid columns */
  .toolbar {display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; margin-bottom:.3rem;}
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {
    "scrollZoom": True,       # wheel zoom (x). Hold SHIFT for vertical zoom
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
}

# =============== Helpers ===============
def parse_tickers(text: str) -> list[str]:
    parts = [t.strip().upper() for t in (text or "").replace(";", ",").split(",") if t.strip()]
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def sma(s: pd.Series, w: int) -> pd.Series: return s.rolling(w, min_periods=w).mean()
def ema(s: pd.Series, w: int) -> pd.Series: return s.ewm(span=w, adjust=False, min_periods=w).mean()

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

def session_for_symbol(sym: str):
    """Infer regular session: .NS => NSE, else US."""
    if sym.endswith(".NS"):
        return ("Asia/Kolkata", dt.time(9, 15), dt.time(15, 30))  # NSE
    return ("America/New_York", dt.time(9, 30), dt.time(16, 0))   # US

def filter_trading_hours(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Keep only weekday regular-session bars (removes overnight gaps)."""
    if df.empty: return df
    tz_name, open_t, close_t = session_for_symbol(sym)
    if getattr(df["Date"].dt, "tz", None) is None:
        df["Date"] = df["Date"].dt.tz_localize("UTC")
    local = df["Date"].dt.tz_convert(tz_name)
    mask = (local.dt.weekday < 5) & (local.dt.time >= open_t) & (local.dt.time <= close_t)
    out = df.loc[mask].copy()
    out["Date"] = local.loc[mask].dt.tz_localize(None)
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_logo(symbol: str) -> str | None:
    """Try yfinance logo; else website -> Clearbit."""
    try:
        info = yf.Ticker(symbol).get_info()
        url = (info or {}).get("logo_url")
        if url: return url
        website = (info or {}).get("website") or (info or {}).get("websiteUrl")
        if website:
            domain = website.split("//")[-1].split("/")[0]
            return f"https://logo.clearbit.com/{domain}"
    except Exception:
        pass
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": symbol, "quotesCount": 1, "newsCount": 0},
                         timeout=8)
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        website = quotes[0].get("website") if quotes else None
        if website:
            domain = website.split("//")[-1].split("/")[0]
            return f"https://logo.clearbit.com/{domain}"
    except Exception:
        pass
    return None

def enhance(fig: go.Figure) -> go.Figure:
    # IMPORTANT: no rangeslider/rangeselector (fully inside chart; mouse zoom/pan only)
    fig.update_layout(
        dragmode="pan", hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0f1116", paper_bgcolor="#0f1116", font=dict(color="#e5e5e5"),
    )
    fig.update_xaxes(fixedrange=False, showgrid=True, gridcolor="rgba(80,80,80,0.25)",
                     rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False, showgrid=True, gridcolor="rgba(80,80,80,0.25)")
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid")
    fig.update_yaxes(showspikes=True, spikemode="across", spikedash="solid")
    return fig

# =============== Data fetch ===============
@st.cache_data(ttl=180, show_spinner=False)
def fetch_prices(tkr: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    """Intraday → last 7d; Daily+ → start/end window."""
    try:
        if interval.endswith("m"):
            df = yf.download(tkr, period="7d", interval=interval, progress=False, threads=False)
        else:
            df = yf.download(tkr, start=start, end=end + dt.timedelta(days=1),
                             interval=interval, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])  # keep tz for filtering
    cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols]

# =============== Sidebar ===============
with st.sidebar:
    st.markdown('<div class="sidebar-title">📈 Stock Tracker</div>', unsafe_allow_html=True)

    tickers_text = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state.get("tickers_input", "AAPL, MSFT, TSLA"),
        help="Examples: AAPL • MSFT • TSLA • RELIANCE.NS • TCS.NS • BTC-USD"
    )
    st.session_state["tickers_input"] = tickers_text

    interval = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"], index=0)

    indicator_opts = st.multiselect(
        "Indicators", ["SMA 20", "SMA 50", "EMA 20", "EMA 50", "RSI 14", "Heikin Ashi"],
        default=["SMA 20", "SMA 50"]
    )

    # Layout toggle
    layout_choice = st.radio("Layout", ["Vertical (stacked)", "Grid (2 columns)"], index=0)

    today = dt.date.today()
    start_date = st.date_input("Start (for 1d/1wk/1mo)", value=today - dt.timedelta(days=365))
    end_date   = st.date_input("End (for 1d/1wk/1mo)",   value=today)

    live_mode   = st.checkbox("🔴 Live mode (auto-refresh 1m)", False)
    refresh_sec = st.number_input("Refresh every (sec)", 5, 120, 15, 5)

tickers = parse_tickers(st.session_state["tickers_input"])
if not tickers:
    st.info("Add at least one ticker in the sidebar."); st.stop()

# =============== Chart builders ===============
def build_price_volume(df: pd.DataFrame, label: str, height: int = 700) -> tuple[go.Figure, pd.Series]:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22], specs=[[{}], [{}]])
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name=label), row=1, col=1)
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume",
                             marker=dict(color="rgba(140,170,255,0.5)")), row=2, col=1)
    fig.update_layout(height=height)
    return fig, df["Close"]

def rsi_panel(series: pd.Series, dates: pd.Series):
    r = rsi(series, 14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=r, mode="lines", name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dot"); fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(title="RSI"), hovermode="x unified",
                      plot_bgcolor="#0f1116", paper_bgcolor="#0f1116", font=dict(color="#e5e5e5"))
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

# =============== Per-chart renderer ===============
def stock_card(symbol: str, compact: bool = False):
    cls = "stock-card compact" if compact else "stock-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)

    # Title with logo + Impact
    logo = get_stock_logo(symbol)
    title_html = f"""
    <div class='toolbar'>
      {'<img src="'+logo+'" style="width:32px;height:32px;border-radius:6px;margin-right:8px;" onerror="this.style.display=\'none\'">' if logo else ''}
      <span style="font-family: Impact, Haettenschweiler, 'Arial Narrow Bold', sans-serif;
                   font-weight:900; font-size:28px; letter-spacing:.5px;">
        {symbol}
      </span>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

    df = fetch_prices(symbol, start_date, end_date, interval)
    if df.empty:
        st.warning("No data for this timeframe/ticker.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if interval.endswith("m"):
        df = filter_trading_hours(df, symbol)
        if df.empty:
            st.warning("No bars in regular session.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    # Heikin Ashi (optional)
    label = symbol
    if "Heikin Ashi" in indicator_opts and len(df) >= 2:
        ha = heikin_ashi(df)
        df = df.copy()
        df["Open"], df["High"], df["Low"], df["Close"] = ha["HA_Open"], ha["HA_High"], ha["HA_Low"], ha["HA_Close"]
        label = f"{symbol} (Heikin Ashi)"

    # Metric
    last = float(df.iloc[-1]["Close"])
    prev = float(df.iloc[-2]["Close"]) if len(df) > 1 else last
    st.metric("Last Price", f"{last:,.2f}", f"{((last - prev) / prev) * 100:+.2f}%")

    # Chart
    fig, close_series = build_price_volume(df, label, height=700)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # weekend skip

    # Overlays
    if "SMA 20" in indicator_opts and len(close_series) >= 20:
        fig.add_trace(go.Scatter(x=df["Date"], y=sma(close_series, 20), mode="lines", name="SMA 20"), row=1, col=1)
    if "SMA 50" in indicator_opts and len(close_series) >= 50:
        fig.add_trace(go.Scatter(x=df["Date"], y=sma(close_series, 50), mode="lines", name="SMA 50"), row=1, col=1)
    if "EMA 20" in indicator_opts and len(close_series) >= 20:
        fig.add_trace(go.Scatter(x=df["Date"], y=ema(close_series, 20), mode="lines", name="EMA 20"), row=1, col=1)
    if "EMA 50" in indicator_opts and len(close_series) >= 50:
        fig.add_trace(go.Scatter(x=df["Date"], y=ema(close_series, 50), mode="lines", name="EMA 50"), row=1, col=1)

    fig = enhance(fig)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    # RSI (optional)
    if "RSI 14" in indicator_opts and len(close_series) >= 15:
        rsi_panel(close_series, df["Date"])

    st.markdown("</div>", unsafe_allow_html=True)

# =============== Render (layout toggle) ===============
if layout_choice.startswith("Vertical"):
    for sym in tickers:
        stock_card(sym, compact=False)
else:
    col1, col2 = st.columns(2, gap="large")
    for i, sym in enumerate(tickers):
        with (col1 if i % 2 == 0 else col2):
            stock_card(sym, compact=True)

# =============== Live refresh ===============
if live_mode:
    ph = st.empty()
    for s in range(int(refresh_sec), 0, -1):
        ph.caption(f"🔴 Live mode — updating in {s} sec… (SHIFT+wheel = vertical zoom)")
        time.sleep(1)
    try: st.rerun()
    except Exception: st.experimental_rerun()
