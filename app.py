import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import find_peaks
from scipy.stats import skew
import seaborn as sns
import matplotlib.ticker as ticker
from pathlib import Path 

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Brent–WTI Dashboards", layout="wide")
st.title("Brent–WTI Dashboards")

# -----------------------------
# Tab Selection
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Default", "Weighted", "Calendar"])

# -----------------------------
# Load Data
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

def resolve_path(path: str | Path) -> Path:
    """
    Resolve file paths reliably across:
    - Local runs
    - Streamlit Cloud
    - VS Code / Dev Containers
    """
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p
@st.cache_data
def load_data(path: str | Path) -> pd.DataFrame:
    path = resolve_path(path)

    if not path.exists():
        st.error(f"Data file not found: {path}")
        st.stop()

    df_base = pd.read_excel(path)

    df_base.columns = [
        "Timestamp",
        "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE",
        "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE",
    ]

    df_base["Timestamp"] = pd.to_datetime(df_base["Timestamp"])
    df_base = df_base.sort_values("Timestamp").reset_index(drop=True)
    return df_base

tab_config = {
    "C1": resolve_path("Brent_WTI_C1.xlsx"),
    "C2": resolve_path("Brent_WTI_C2.xlsx"),
    "C3": resolve_path("Brent_WTI_C3.xlsx"),
}

def add_spreads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread_close"] = df["WTI_CLOSE"] - df["Brent_CLOSE"]
    df["spread_high"] = df["WTI_HIGH"] - df["Brent_HIGH"]
    df["spread_WTI_low_Brent_high"] = df["WTI_LOW"] - df["Brent_HIGH"]
    df["spread_WTI_high_Brent_low"] = df["WTI_HIGH"] - df["Brent_LOW"]
    return df

def build_spread_series(df: pd.DataFrame, spread_key: str) -> pd.Series:
    ts = pd.to_datetime(df["Timestamp"])
    s = df[spread_key].copy()
    s.index = ts
    return s.sort_index()

# =========================================================
# 1) Spread Dashboard
# =========================================================
def render_spread_dashboard_streamlit(lookback, ma_window, visual_choice, show_ma, show_ma_sd, df_base):
    import matplotlib.transforms as mtransforms

    latest_date = df_base["Timestamp"].max()
    cutoff_date = latest_date - pd.DateOffset(years=int(lookback))
    df = df_base[df_base["Timestamp"] >= cutoff_date].copy()

    # Calculate Spreadspip
    df["spread_close"] = df["WTI_CLOSE"] - df["Brent_CLOSE"]
    df["spread_high"] = df["WTI_HIGH"] - df["Brent_HIGH"]
    df["spread_WTI_low_Brent_high"] = df["WTI_LOW"] - df["Brent_HIGH"]
    df["spread_WTI_high_Brent_low"] = df["WTI_HIGH"] - df["Brent_LOW"]

    labels_map = {
        "spread_close": ("Close-to-Close", "#2E86C1"),
        "spread_high": ("High-to-High", "#27AE60"),
        "spread_WTI_low_Brent_high": ("Floor (WTI Low - Brent High)", "#E74C3C"),
        "spread_WTI_high_Brent_low": ("Ceiling (WTI High - Brent Low)", "#8E44AD"),
    }
    label, color = labels_map[visual_choice]

    s = build_spread_series(df, visual_choice)

    plt.close("all")
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.plot(s.index, s.values, label=label, color=color, lw=1.8, alpha=0.9, zorder=4)

    trend_w, trend_m = "N/A", "N/A"
    ma_stats_text = ""

    if show_ma:
        safe_window = max(2, min(int(ma_window), len(s)))
        ma_line = s.rolling(window=safe_window).mean()
        ax.plot(
            ma_line.index, ma_line.values,
            label=f"{safe_window}D MA", color="black",
            lw=1.2, alpha=0.8, zorder=3
        )

        curr_sma = ma_line.iloc[-1]
        week_sma = ma_line.iloc[-6] if len(ma_line) > 6 else np.nan
        month_sma = ma_line.iloc[-22] if len(ma_line) > 22 else np.nan
        trend_w = "UP" if curr_sma > week_sma else "DOWN"
        trend_m = "UP" if curr_sma > month_sma else "DOWN"

        if show_ma_sd:
            ma_sd = s.rolling(window=safe_window).std()
            u1, u2 = ma_line + ma_sd, ma_line + 2 * ma_sd
            l1, l2 = ma_line - ma_sd, ma_line - 2 * ma_sd
            ma_stats_text = (
                f"\nMA ±1SD: {l1.iloc[-1]:.2f} / {u1.iloc[-1]:.2f}"
                f"\nMA ±2SD: {l2.iloc[-1]:.2f} / {u2.iloc[-1]:.2f}"
            )
            ax.fill_between(s.index, l1.values, u1.values, color="#1ABC9C", alpha=0.15, label="MA ±1SD", zorder=2)
            ax.fill_between(s.index, l2.values, l1.values, color="#9B59B6", alpha=0.10, label="MA ±2SD", zorder=1)
            ax.fill_between(s.index, u1.values, u2.values, color="#9B59B6", alpha=0.10, zorder=1)

    ref_mean, ref_std = float(s.mean()), float(s.std())

    ax.axhline(ref_mean, color="black", lw=2, alpha=0.4, label="Global Mean")
    ax.axhline(ref_mean + ref_std, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axhline(ref_mean - ref_std, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axhline(ref_mean + 2 * ref_std, color="#C0392B", lw=1.5, ls=":", alpha=0.7)
    ax.axhline(ref_mean - 2 * ref_std, color="#C0392B", lw=1.5, ls=":", alpha=0.7)

    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    line_labels = [
        (ref_mean + 2 * ref_std, f"+2σ  {ref_mean + 2 * ref_std:.2f}"),
        (ref_mean + 1 * ref_std, f"+1σ  {ref_mean + 1 * ref_std:.2f}"),
        (ref_mean,               f"μ    {ref_mean:.2f}"),
        (ref_mean - 1 * ref_std, f"-1σ  {ref_mean - 1 * ref_std:.2f}"),
        (ref_mean - 2 * ref_std, f"-2σ  {ref_mean - 2 * ref_std:.2f}"),
    ]
    for y, txt in line_labels:
        ax.text(
            0.995, y, txt,
            transform=trans,
            ha="right", va="center",
            fontsize=8, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.5),
            zorder=10,
            clip_on=False,
        )

    current_val, current_date = float(s.iloc[-1]), s.index[-1]
    z_score = (current_val - ref_mean) / ref_std if ref_std != 0 else np.nan

    if ref_std != 0 and not np.isnan(z_score):
        lower_sd_limit = np.floor(z_score)
        upper_sd_limit = lower_sd_limit + 1
        z_series = (s - ref_mean) / ref_std
        in_range_mask = z_series.between(lower_sd_limit, upper_sd_limit, inclusive="left")
        persistence_pct = float(in_range_mask.mean() * 100)
    else:
        lower_sd_limit, upper_sd_limit, persistence_pct = 0, 1, 0.0

    curr_xlim = ax.get_xlim()
    ax.set_xlim(curr_xlim[0], curr_xlim[1] + (curr_xlim[1] - curr_xlim[0]) * 0.25)

    ax.scatter(current_date, current_val, color="blue", s=60, zorder=6)

    y_frac = 0.85 if current_val < ref_mean else 0.15
    now_text = (
        f"Current Spread: {current_val:.2f} ({z_score:+.1f} SD)\n"
        f"Range Freq: {persistence_pct:.1f}%"
        f"{ma_stats_text}"
    )
    ax.annotate(
        now_text,
        xy=(current_date, current_val), xycoords="data",
        xytext=(1.02, y_frac), textcoords="axes fraction",
        ha="left", va="center",
        fontsize=9, color="blue", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, boxstyle="round,pad=0.35"),
        arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        annotation_clip=False,
    )

    ax.text(
        0.5, 0.98, f"SMA Trend: [Weekly: {trend_w}] | [Monthly: {trend_m}]",
        transform=ax.transAxes, ha="center", va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.5"),
        fontsize=10, fontweight="bold",
    )

    ax.set_title(f"1) Spread Dashboard | {int(lookback)}Y History | {label}", loc="left", fontsize=12, pad=25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.1)

    plt.tight_layout()

    st.subheader("1) Spread Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current spread", f"{current_val:.2f}")
    c2.metric("Range Freq", f"{persistence_pct:.1f}%")
    c3.metric("Weekly trend (MA)", trend_w)
    c4.metric("Monthly trend (MA)", trend_m)
    st.pyplot(fig, use_container_width=True)

# =========================================================
# 2) MA Retracement Dashboard 
# =========================================================
def render_ma_retracement_dashboard_streamlit(
    lookback, ma_window, visual_choice,
    min_gap_days=20,
    retrace_levels=(0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10),
    max_forward_days=250,
    df_base=None
):
    latest_date = df_base["Timestamp"].max()
    cutoff_date = latest_date - pd.DateOffset(years=int(lookback))
    df = df_base[df_base["Timestamp"] >= cutoff_date].copy()
    df = add_spreads(df)

    labels_map = {
        "spread_close": ("Close-to-Close", "#2E86C1"),
        "spread_high": ("High-to-High", "#27AE60"),
        "spread_WTI_low_Brent_high": ("Floor (WTI Low - Brent High)", "#E74C3C"),
        "spread_WTI_high_Brent_low": ("Ceiling (WTI High - Brent Low)", "#8E44AD"),
    }
    label, _ = labels_map[visual_choice]

    s = build_spread_series(df, visual_choice)
    safe_window = max(2, min(int(ma_window), len(s)))
    ma = s.rolling(window=safe_window).mean()
    amp = (s - ma).dropna()

    # --- Peaks/troughs on amplitude (SciPy robust) ---
    noise_threshold = float(amp.std() * 0.15)
    peaks, _ = find_peaks(amp.values, distance=int(min_gap_days), prominence=noise_threshold)
    troughs, _ = find_peaks(-amp.values, distance=int(min_gap_days), prominence=noise_threshold)

    inflexions = sorted(
        [(amp.index[i], "peak") for i in peaks] +
        [(amp.index[i], "trough") for i in troughs],
        key=lambda x: x[0]
    )

    # --- Snapshot-based detection for historical grey arrows ---
    confirmation_days = 5
    check_interval = 1  # Check every day
    historical_detections = []  # Store superseded "latest" inflections
    
    last_peak_detected = None
    last_trough_detected = None
    
    for i in range(safe_window + confirmation_days, len(amp), check_interval):
        # Use data up to current point plus confirmation window
        end_idx = min(i + confirmation_days, len(amp))
        rolling_amp = amp.iloc[:end_idx]
        
        # Calculate rolling noise threshold
        rolling_noise = float(rolling_amp.std() * 0.15)
        
        # Detect peaks/troughs in this rolling window
        roll_peaks, _ = find_peaks(rolling_amp.values, distance=int(min_gap_days), prominence=rolling_noise)
        roll_troughs, _ = find_peaks(-rolling_amp.values, distance=int(min_gap_days), prominence=rolling_noise)
        
        # Find the latest peak and trough in this window (confirmed ones only)
        confirmed_peaks = [idx for idx in roll_peaks if idx < len(rolling_amp) - confirmation_days]
        confirmed_troughs = [idx for idx in roll_troughs if idx < len(rolling_amp) - confirmation_days]
        
        current_peak = rolling_amp.index[confirmed_peaks[-1]] if confirmed_peaks else None
        current_trough = rolling_amp.index[confirmed_troughs[-1]] if confirmed_troughs else None
        
        # Track evolution of "latest peak"
        if current_peak is not None:
            if last_peak_detected is not None and current_peak != last_peak_detected:
                # The "latest peak" has changed - archive the old one as grey
                if last_peak_detected not in [d for d, _ in historical_detections]:
                    historical_detections.append((last_peak_detected, "peak"))
            last_peak_detected = current_peak
        
        # Track evolution of "latest trough"
        if current_trough is not None:
            if last_trough_detected is not None and current_trough != last_trough_detected:
                # The "latest trough" has changed - archive the old one as grey
                if last_trough_detected not in [d for d, _ in historical_detections]:
                    historical_detections.append((last_trough_detected, "trough"))
            last_trough_detected = current_trough
    
    historical_detections.sort(key=lambda x: x[0])

    # --- Sensitivity table with FIXED retracement logic ---
    summary_rows = []
    for lvl in list(retrace_levels):
        days_list, move_list = [], []
        for t0, _ in inflexions:
            a0 = amp.loc[t0]
            if pd.isna(a0) or a0 == 0:
                continue
            future = amp.loc[t0:].iloc[1:max_forward_days + 1]
            for d, (t1, a1) in enumerate(future.items(), start=1):
                if pd.isna(a1):
                    continue
                
                # Check if retraced to target level
                if abs(float(a1)) <= float(lvl) * abs(float(a0)):
                    days_list.append(d)
                    move_list.append(abs(float(s.loc[t1]) - float(s.loc[t0])))
                    break
                
                #  Also check if amplitude crossed zero (hit the MA)
                if (a0 > 0 and a1 < 0) or (a0 < 0 and a1 > 0):
                    days_list.append(d)
                    move_list.append(abs(float(s.loc[t1]) - float(s.loc[t0])))
                    break

        retrace_pct = int((1 - float(lvl)) * 100)
        avg_days = float(np.mean(days_list)) if days_list else 0.0
        std_days = float(np.std(days_list)) if days_list else 0.0
        avg_move = float(np.mean(move_list)) if move_list else 0.0
        summary_rows.append([f"{retrace_pct}%", f"{avg_days:.1f}", f"±{std_days:.1f}", f"${avg_move:.2f}"])

    # --- Plot ---
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(right=0.78, hspace=0.10)

    ax1.plot(s.index, s.values, label=label, alpha=0.55, zorder=1)
    ax1.plot(ma.index, ma.values, label=f"{safe_window}D MA", color="black", lw=1.8, zorder=2)
    ax1.set_ylabel("Spread ($/bbl)")

    ax2.plot(amp.index, amp.values, label="MA Amplitude (Spread − MA)", lw=1.6, zorder=2)
    ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax2.set_ylabel("Amplitude ($)")

    peak_dates = [t for t, typ in inflexions if typ == "peak"]
    trough_dates = [t for t, typ in inflexions if typ == "trough"]

    # Plot historical detections (grey arrows)
    hist_peak_dates = [t for t, typ in historical_detections if typ == "peak"]
    hist_trough_dates = [t for t, typ in historical_detections if typ == "trough"]
    
    if hist_peak_dates:
        ax2.scatter(hist_peak_dates, amp.loc[hist_peak_dates], marker="v", color="grey", s=40, zorder=4, alpha=0.35, label="Historical Peak")
        ax1.scatter(hist_peak_dates, s.loc[hist_peak_dates], marker="v", color="grey", s=30, zorder=4, alpha=0.35)
    if hist_trough_dates:
        ax2.scatter(hist_trough_dates, amp.loc[hist_trough_dates], marker="^", color="grey", s=40, zorder=4, alpha=0.35, label="Historical Trough")
        ax1.scatter(hist_trough_dates, s.loc[hist_trough_dates], marker="^", color="grey", s=30, zorder=4, alpha=0.35)

    # Plot current inflections (colored arrows)
    if peak_dates:
        ax2.scatter(peak_dates, amp.loc[peak_dates], marker="v", color="blue", s=45, zorder=5, label="Peak")
        ax1.scatter(peak_dates, s.loc[peak_dates], marker="v", color="blue", s=35, zorder=5)
    if trough_dates:
        ax2.scatter(trough_dates, amp.loc[trough_dates], marker="^", color="red", s=45, zorder=5, label="Trough")
        ax1.scatter(trough_dates, s.loc[trough_dates], marker="^", color="red", s=35, zorder=5)

    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper left", fontsize=9)

    # --- Table: Latest start + retrace prices (top-right) ---
    typ_lat, t_lat = None, None
    if inflexions:
        t_lat, typ_lat = inflexions[-1]
        a0_lat = amp.loc[t_lat]

        inflex_data = [
            ["Type", typ_lat.upper()],
            ["Date", pd.Timestamp(t_lat).strftime("%Y-%m-%d")],
            ["MA Window (D)", f"{safe_window}"],
            ["Spread @ Start", f"${float(s.loc[t_lat]):.2f}"],
            ["MA @ Start", f"${float(ma.loc[t_lat]):.2f}"],
            ["Amp @ Start", f"${float(a0_lat):.2f}"],
        ]

        future_lat = amp.loc[t_lat:].iloc[1:max_forward_days + 1]
        for lvl in retrace_levels:
            retrace_pct = int((1 - float(lvl)) * 100)
            target_display = "Pending"
            for t1, a1 in future_lat.items():
                if pd.isna(a1):
                    continue
                
                # Check if retraced to target level
                if abs(float(a1)) <= float(lvl) * abs(float(a0_lat)):
                    target_display = f"${float(s.loc[t1]):.2f} ({pd.Timestamp(t1).strftime('%Y-%m-%d')})"
                    break
                
                # Also check if amplitude crossed zero (hit the MA)
                if (a0_lat > 0 and a1 < 0) or (a0_lat < 0 and a1 > 0):
                    target_display = f"${float(s.loc[t1]):.2f} ({pd.Timestamp(t1).strftime('%Y-%m-%d')})"
                    break
                    
            inflex_data.append([f"{retrace_pct}% Level", target_display])

        inflex_ax = fig.add_axes([0.80, 0.58, 0.19, 0.37])
        inflex_ax.axis("off")
        inflex_ax.set_title("LATEST MA-RETRACE\nSTART & RETRACE PRICES", fontsize=9, fontweight="bold")
        t1 = inflex_ax.table(cellText=inflex_data, loc="center", cellLoc="left")
        t1.auto_set_font_size(False)
        t1.set_fontsize(7.5)
        t1.scale(1.0, 1.35)

    # --- Table: Sensitivity (bottom-right) ---
    table_ax = fig.add_axes([0.80, 0.08, 0.19, 0.45])
    table_ax.axis("off")
    table_ax.text(0.5, 0.98, "MA Retrace Sensitivity", ha="center", va="top",
                  fontsize=9, fontweight="bold", transform=table_ax.transAxes)

    tbl = table_ax.table(
        cellText=summary_rows,
        colLabels=["Retrace %", "Avg Days", "1 SD", "Avg $ Move"],
        loc="center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.90]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.25)

    # --- Streamlit header + metrics + interpretation ---
    st.subheader("2) MA Retracement Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MA Window (D)", int(safe_window))
    c2.metric("# Inflexions", int(len(inflexions)))
    if inflexions:
        c3.metric("Last Start", typ_lat.upper())
        c4.metric("Last Start Date", pd.Timestamp(t_lat).strftime("%Y-%m-%d"))
    else:
        c3.metric("Last Start", "N/A")
        c4.metric("Last Start Date", "N/A")

    st.pyplot(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.write("**Amplitude = Spread − MA.** Peaks/troughs are detected on the amplitude series.")
    st.write("Retracement timing measures **how many days until |Amplitude| shrinks to X% of its starting value OR crosses the MA**.")
    st.write("The sensitivity table aggregates retracement behavior across all detected inflexions in the lookback window.")

# =========================================================
# 3) Persistence Dashboard 
# =========================================================
def render_persistence_dashboard_streamlit(lookback, fast_ma, slow_ma, df_base):
    latest_date = df_base["Timestamp"].max()
    cutoff_date = latest_date - pd.DateOffset(years=int(lookback))
    df = df_base[df_base["Timestamp"] >= cutoff_date].copy()

    s = (df["WTI_CLOSE"] - df["Brent_CLOSE"]).copy()
    s.index = df["Timestamp"]
    s = s.sort_index().dropna()

    if int(fast_ma) >= int(slow_ma):
        st.error("Error: Fast MA must be smaller than Slow MA.")
        return

    ma_fast = s.rolling(int(fast_ma)).mean()
    ma_slow = s.rolling(int(slow_ma)).mean()
    signal = (ma_fast - ma_slow).dropna()

    # Peaks/troughs on momentum signal
    p_noise = float(signal.std() * 0.15)
    sig_peaks, _ = find_peaks(signal.values, distance=20, prominence=p_noise)
    sig_troughs, _ = find_peaks(-signal.values, distance=20, prominence=p_noise)

    inflexions = sorted(
        [(signal.index[i], "peak") for i in sig_peaks] +
        [(signal.index[i], "trough") for i in sig_troughs],
        key=lambda x: x[0]
    )

    # --- Rolling detection for historical grey arrows ---
    confirmation_days = 10
    check_interval = 5  # Check every 5 days
    historical_detections = []  # Store superseded "latest" inflections
    
    last_peak_detected = None
    last_trough_detected = None
    
    min_window = max(int(fast_ma), int(slow_ma))
    for i in range(min_window + confirmation_days, len(signal), check_interval):
        # Use data up to current point plus confirmation window
        end_idx = min(i + confirmation_days, len(signal))
        rolling_signal = signal.iloc[:end_idx]
        
        # Calculate rolling noise threshold
        rolling_noise = float(rolling_signal.std() * 0.15)
        
        # Detect peaks/troughs in this rolling window
        roll_peaks, _ = find_peaks(rolling_signal.values, distance=20, prominence=rolling_noise)
        roll_troughs, _ = find_peaks(-rolling_signal.values, distance=20, prominence=rolling_noise)
        
        # Find the latest peak and trough in this window (confirmed ones only)
        confirmed_peaks = [idx for idx in roll_peaks if idx < len(rolling_signal) - confirmation_days]
        confirmed_troughs = [idx for idx in roll_troughs if idx < len(rolling_signal) - confirmation_days]
        
        current_peak = rolling_signal.index[confirmed_peaks[-1]] if confirmed_peaks else None
        current_trough = rolling_signal.index[confirmed_troughs[-1]] if confirmed_troughs else None
        
        # Track evolution of "latest peak"
        if current_peak is not None:
            if last_peak_detected is not None and current_peak != last_peak_detected:
                # The "latest peak" has changed - archive the old one as grey
                if last_peak_detected not in [d for d, _ in historical_detections]:
                    historical_detections.append((last_peak_detected, "peak"))
            last_peak_detected = current_peak
        
        # Track evolution of "latest trough"
        if current_trough is not None:
            if last_trough_detected is not None and current_trough != last_trough_detected:
                # The "latest trough" has changed - archive the old one as grey
                if last_trough_detected not in [d for d, _ in historical_detections]:
                    historical_detections.append((last_trough_detected, "trough"))
            last_trough_detected = current_trough
    
    historical_detections.sort(key=lambda x: x[0])

    decay_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    summary_rows = []
    for d_frac in decay_values:
        wane_days, wane_diffs = [], []
        for t0, _ in inflexions:
            s0 = signal.loc[t0]
            if pd.isna(s0) or s0 == 0:
                continue
            future = signal.loc[t0:].iloc[1:250]
            for d, (t1, val) in enumerate(future.items(), start=1):
                if pd.isna(val):
                    continue
                if abs(float(val)) <= float(d_frac) * abs(float(s0)):
                    wane_days.append(d)
                    wane_diffs.append(abs(float(s.loc[t1]) - float(s.loc[t0])))
                    break

        retrace_pct = int((1 - float(d_frac)) * 100)
        avg_days = float(np.mean(wane_days)) if wane_days else 0.0
        std_days = float(np.std(wane_days)) if wane_days else 0.0
        avg_move = float(np.mean(wane_diffs)) if wane_diffs else 0.0
        summary_rows.append([f"{retrace_pct}%", f"{avg_days:.1f}", f"±{std_days:.1f}", f"${avg_move:.2f}"])

    # Plot
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(right=0.78, hspace=0.10)

    ax1.plot(s.index, s.values, label="Actual Spread", alpha=0.5, zorder=1)
    ax1.plot(ma_fast.index, ma_fast.values, label=f"Fast {int(fast_ma)}D", lw=2, zorder=3)
    ax1.plot(ma_slow.index, ma_slow.values, label=f"Slow {int(slow_ma)}D", lw=2, zorder=2)
    ax1.set_ylabel("Spread ($/bbl)")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(signal.index, signal.values, label="Momentum Signal (Fast MA − Slow MA)", lw=1.5)
    ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax2.set_ylabel("Signal")

    peak_dates = [t for t, typ in inflexions if typ == "peak"]
    trough_dates = [t for t, typ in inflexions if typ == "trough"]

    # Plot historical detections 
    hist_peak_dates = [t for t, typ in historical_detections if typ == "peak"]
    hist_trough_dates = [t for t, typ in historical_detections if typ == "trough"]
    
    if hist_peak_dates:
        ax2.scatter(hist_peak_dates, signal.loc[hist_peak_dates], marker="v", color="grey", s=90, zorder=4, alpha=0.35, label="Historical Peak")
    if hist_trough_dates:
        ax2.scatter(hist_trough_dates, signal.loc[hist_trough_dates], marker="^", color="grey", s=90, zorder=4, alpha=0.35, label="Historical Trough")

    # Plot current inflections (colored arrows)
    if peak_dates:
        ax2.scatter(peak_dates, signal.loc[peak_dates], marker="v", color="red", s=100, label="Peak", zorder=5)
    if trough_dates:
        ax2.scatter(trough_dates, signal.loc[trough_dates], marker="^", color="blue", s=100, label="Trough", zorder=5)

    ax2.legend(loc="upper left", fontsize=9)

    # Latest start + decay targets (top-right)
    typ_lat, t_lat = None, None
    target_levels = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    max_forward_days = 300

    if inflexions:
        t_lat, typ_lat = inflexions[-1]
        s0_lat = signal.loc[t_lat]

        inflex_data = [
            ["Type", typ_lat.upper()],
            ["Date", pd.Timestamp(t_lat).strftime("%Y-%m-%d")],
            ["Fast MA (D)", f"{int(fast_ma)}"],
            ["Slow MA (D)", f"{int(slow_ma)}"],
            ["Spread @ Start", f"${float(s.loc[t_lat]):.2f}"],
            ["Signal @ Start", f"{float(s0_lat):.2f}"],
        ]

        future_lat = signal.loc[t_lat:].iloc[1:max_forward_days + 1]
        for lvl in target_levels:
            retrace_pct = int((1 - float(lvl)) * 100)
            target_display = "Pending"
            for t1, v1 in future_lat.items():
                if pd.isna(v1):
                    continue
                if abs(float(v1)) <= float(lvl) * abs(float(s0_lat)):
                    target_display = f"${float(s.loc[t1]):.2f} ({pd.Timestamp(t1).strftime('%Y-%m-%d')})"
                    break
            inflex_data.append([f"{retrace_pct}% Level", target_display])

        inflex_ax = fig.add_axes([0.80, 0.58, 0.19, 0.37])
        inflex_ax.axis("off")
        inflex_ax.set_title("LATEST SIGNAL\nSTART & RETRACE PRICES", fontsize=9, fontweight="bold")
        t1 = inflex_ax.table(cellText=inflex_data, loc="center", cellLoc="left")
        t1.auto_set_font_size(False)
        t1.set_fontsize(7.5)
        t1.scale(1.0, 1.35)

    # Sensitivity table (bottom-right)
    table_ax = fig.add_axes([0.80, 0.08, 0.19, 0.45])
    table_ax.axis("off")
    table_ax.text(0.5, 0.98, "Persistence Sensitivity", ha="center", va="top",
                  fontsize=9, fontweight="bold", transform=table_ax.transAxes)

    tbl = table_ax.table(
        cellText=summary_rows,
        colLabels=["Retrace %", "Avg Days", "1 SD", "Avg $ Move"],
        loc="center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.90]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.25)

    # Streamlit header + metrics
    st.subheader("3) Persistence Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fast MA", int(fast_ma))
    c2.metric("Slow MA", int(slow_ma))
    c3.metric("# Inflexions", int(len(inflexions)))
    if inflexions:
        c4.metric("Last Signal", typ_lat.upper(), delta=pd.Timestamp(t_lat).strftime("%Y-%m-%d"), delta_color="off")
    else:
        c4.metric("Last Signal", "N/A")

    st.pyplot(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.write("The **momentum signal = Fast MA − Slow MA**. Peaks/troughs are detected on this signal.")
    st.write("The sensitivity table measures **how many days until |Signal| shrinks to X% of its starting value** (persistence).")

# =========================================================
# UNIFIED SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Dashboard Controls")

st.sidebar.subheader("1) Spread Dashboard Controls")
lookback = st.sidebar.number_input("Lookback (Y)", min_value=1, max_value=30, value=3, step=1, key="lookback")
ma_window = st.sidebar.number_input("MA Window (D)", min_value=5, max_value=400, value=50, step=5, key="ma_window")

view_map = {
    "Close": "spread_close",
    "High": "spread_high",
    "Floor": "spread_WTI_low_Brent_high",
    "Ceiling": "spread_WTI_high_Brent_low",
}
visual_label = st.sidebar.selectbox("View:", options=list(view_map.keys()), index=0, key="view")
visual_choice = view_map[visual_label]

show_ma = st.sidebar.checkbox("Show MA", value=True, key="show_ma")
show_ma_sd = st.sidebar.checkbox("Show MA Bands", value=True, key="show_ma_sd")

st.sidebar.divider()
st.sidebar.subheader("2) MA Retracement Dashboard Controls")
mr_lookback = st.sidebar.number_input("MA Retracement Lookback (Y)", min_value=1, max_value=30, value=3, step=1, key="mr_lookback")
mr_ma_window = st.sidebar.number_input("MA Window for Retracement (D)", min_value=5, max_value=400, value=int(ma_window), step=5, key="mr_ma_window")

st.sidebar.divider()
st.sidebar.subheader("3) Persistence Dashboard Controls")
p_lookback = st.sidebar.number_input("Persistence Lookback (Y)", min_value=1, max_value=30, value=3, step=1, key="p_lookback")
fast_ma = st.sidebar.number_input("Fast MA Window", min_value=2, max_value=300, value=20, step=1, key="fast_ma")
slow_ma = st.sidebar.number_input("Slow MA Window", min_value=5, max_value=600, value=50, step=1, key="slow_ma")

# Fixed params
MR_MIN_GAP_DAYS = 20
MR_RETRACE_LEVELS = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

# =========================================================
# TAB 1: DEFAULT
# =========================================================
with tab1:
    contract_col1, contract_col2 = st.columns([1, 6])
    with contract_col1:
        selected_contract = st.selectbox(
            "Contract",
            options=["C1", "C2", "C3"],
            index=0,
            key="default_contract"
        )

    DATA_PATH = tab_config[selected_contract]
    df_base = load_data(DATA_PATH)

    # Render dashboards using unified sidebar controls
    render_spread_dashboard_streamlit(lookback, ma_window, visual_choice, show_ma, show_ma_sd, df_base)
    st.divider()
    render_ma_retracement_dashboard_streamlit(
        mr_lookback,
        int(mr_ma_window),
        visual_choice,
        MR_MIN_GAP_DAYS,
        MR_RETRACE_LEVELS,
        df_base=df_base
    )
    st.divider()
    render_persistence_dashboard_streamlit(p_lookback, fast_ma, slow_ma, df_base)


# =========================================================
# Business-day helpers 
# =========================================================
def bdays_to_expiry(d: pd.Timestamp, expiry: pd.Timestamp) -> int:
    """
    Business days remaining INCLUDING expiry day.
    So if d == expiry (and it's a business day), returns 1.
    If d > expiry, returns 0.
    """
    d = pd.to_datetime(d).normalize()
    expiry = pd.to_datetime(expiry).normalize()

    if d > expiry:
        return 0

    # np.busday_count counts business days in [start, end)
    # so use end = expiry + 1 calendar day to include expiry day
    end = (expiry + pd.Timedelta(days=1)).date()
    return int(np.busday_count(d.date(), end))

def last_business_day_of_month(d: pd.Timestamp) -> pd.Timestamp:
    """Last business day of d's month (weekday-only)."""
    # go to month end, then step back to weekday
    month_end = (d + pd.offsets.MonthEnd(0)).normalize()
    while month_end.weekday() >= 5:  # Sat/Sun
        month_end -= pd.Timedelta(days=1)
    return month_end

def add_business_days(d: pd.Timestamp, n: int) -> pd.Timestamp:
    """Add n business days to date d (weekday-only)."""
    # using pandas BDay
    return (d + pd.offsets.BDay(n)).normalize()

def subtract_business_days(d: pd.Timestamp, n: int) -> pd.Timestamp:
    """Subtract n business days from date d (weekday-only)."""
    return (d - pd.offsets.BDay(n)).normalize()

# =========================================================
# Expiry model 
# =========================================================
def brent_c1_expiry(d: pd.Timestamp) -> pd.Timestamp:
    """
    Brent C1 expiry = last trading day of month (weekday-only approximation).
    """
    return last_business_day_of_month(d)

def brent_c2_expiry(d: pd.Timestamp) -> pd.Timestamp:
    """Brent C2 expiry ~ last business day of next month."""
    return last_business_day_of_month(d + pd.offsets.MonthBegin(1))

def brent_c3_expiry(d: pd.Timestamp) -> pd.Timestamp:
    """Brent C3 expiry ~ last business day of month after next."""
    return last_business_day_of_month(d + pd.offsets.MonthBegin(2))

def wti_c1_expiry(d: pd.Timestamp) -> pd.Timestamp:
    """
    WTI C1 expiry:
    3 trading days before the 25th of the month
    preceding the delivery month.

    Assumption:
    - WTI front-month delivery = d + 2 calendar months
    """
    delivery_month = (d + pd.offsets.MonthBegin(2)).normalize()
    expiry_ref = delivery_month - pd.offsets.MonthBegin(1)  # preceding month
    day25 = pd.Timestamp(
        year=expiry_ref.year,
        month=expiry_ref.month,
        day=25
    )
    return subtract_business_days(day25, 3)


def wti_c2_expiry(d: pd.Timestamp) -> pd.Timestamp:
    """
    WTI C2 expiry = same rule, one month further out
    """
    delivery_month = (d + pd.offsets.MonthBegin(3)).normalize()
    expiry_ref = delivery_month - pd.offsets.MonthBegin(1)
    day25 = pd.Timestamp(
        year=expiry_ref.year,
        month=expiry_ref.month,
        day=25
    )
    return subtract_business_days(day25, 3)

# =========================================================
# Solve integer Brent allocation given fixed W lots and B1 schedule
# =========================================================
def solve_two_tenor_integer(total_brent: int, Tb1: int, Tb2: int, target_exposure: int):
    """
    Solve b1 + b2 = total_brent
          b1*Tb1 + b2*Tb2 = target_exposure
    Return (b1, b2) integers.
    """
    if Tb2 == Tb1:
        # Degenerate: just split
        return total_brent, 0

    # b2 = (target - total*Tb1)/(Tb2-Tb1)
    b2 = (target_exposure - total_brent * Tb1) / (Tb2 - Tb1)
    b2 = int(np.round(b2))
    b2 = int(np.clip(b2, 0, total_brent))
    b1 = total_brent - b2
    return b1, b2

def solve_b2_b3_given_b1_integer(total_brent: int, b1: int, Tb1: int, Tb2: int, Tb3: int, target_exposure: int):
    """
    Given b1 is fixed, solve:
      b2 + b3 = total_brent - b1
      b1*Tb1 + b2*Tb2 + b3*Tb3 = target_exposure

    Return integer (b2, b3).
    """
    rem = total_brent - b1
    if rem < 0:
        return 0, 0

    # Need: b2*Tb2 + b3*Tb3 = target_exposure - b1*Tb1
    rhs = target_exposure - b1 * Tb1

    if Tb3 == Tb2:
        # Degenerate: allocate all to B2
        b3 = 0
        b2 = rem
        return b2, b3

    # Substitute b2 = rem - b3:
    # (rem - b3)*Tb2 + b3*Tb3 = rhs
    # rem*Tb2 + b3*(Tb3 - Tb2) = rhs
    b3 = (rhs - rem * Tb2) / (Tb3 - Tb2)
    b3 = int(np.round(b3))
    b3 = int(np.clip(b3, 0, rem))
    b2 = rem - b3
    return b2, b3

# =========================================================
# Main: build daily lots schedule using your rollover logic
# =========================================================
def build_lots_schedule(dates: pd.Series, total_brent: int, total_wti: int,
                        roll_days: int = 5):
    """
    For each date:
    - compute Tb1,Tb2,Tb3,Tw1,Tw2 via calendar proxies
    - For each MONTH:
        * on the first available trading date in that month: compute baseline B1/B2 (B3=0), W1=total_wti
        * HOLD baseline until rollover window starts (T-roll_days ... T-1 relative to Brent C1 expiry)
        * During rollover window:
            - each day: roll W1 -> W2 by total_wti/roll_days lots per day (must divide evenly)
            - reduce B1 by baseline_B1/roll_days lots per day (assume baseline_B1==roll_days in your example; we implement general)
            - recompute B2/B3 each day to match time exposure
        * After Brent expiry: relabel next month naturally via using new month baseline
    Returns DataFrame with columns:
      Timestamp, Tb1,Tb2,Tb3,Tw1,Tw2, B1,B2,B3,W1,W2
    """
    df = pd.DataFrame({"Timestamp": pd.to_datetime(dates).dt.normalize()}).drop_duplicates().sort_values("Timestamp")
    df["month"] = df["Timestamp"].dt.to_period("M")

    # compute expiry dates & time-to-expiry (weekday-only)
    df["B1_exp"] = df["Timestamp"].apply(brent_c1_expiry)
    df["B2_exp"] = df["Timestamp"].apply(brent_c2_expiry)
    df["B3_exp"] = df["Timestamp"].apply(brent_c3_expiry)
    df["W1_exp"] = df["Timestamp"].apply(wti_c1_expiry)
    df["W2_exp"] = df["Timestamp"].apply(wti_c2_expiry)

    df["Tb1"] = df.apply(lambda r: bdays_to_expiry(r["Timestamp"], r["B1_exp"]), axis=1)
    df["Tb2"] = df.apply(lambda r: bdays_to_expiry(r["Timestamp"], r["B2_exp"]), axis=1)
    df["Tb3"] = df.apply(lambda r: bdays_to_expiry(r["Timestamp"], r["B3_exp"]), axis=1)
    df["Tw1"] = df.apply(lambda r: bdays_to_expiry(r["Timestamp"], r["W1_exp"]), axis=1)
    df["Tw2"] = df.apply(lambda r: bdays_to_expiry(r["Timestamp"], r["W2_exp"]), axis=1)


    # allocate lots columns
    df[["B1","B2","B3","W1","W2"]] = 0

    # sanity: require clean integer daily roll for WTI
    if total_wti % roll_days != 0:
        raise ValueError(f"total_wti must be divisible by roll_days. Got total_wti={total_wti}, roll_days={roll_days}")
    w_roll_per_day = total_wti // roll_days

    # per-month logic
    for m, g in df.groupby("month", sort=True):
        idx = g.index
        if len(idx) == 0:
            continue

        # baseline computed on first trading date we have for that month in the dataset
        first_i = idx[0]
        Tb1_0 = int(df.loc[first_i, "Tb1"])
        Tb2_0 = int(df.loc[first_i, "Tb2"])
        Tw1_0 = int(df.loc[first_i, "Tw1"])

        target0 = total_wti * Tw1_0

        # baseline: two-tenor Brent (B3=0), W1=total_wti
        b1_0, b2_0 = solve_two_tenor_integer(total_brent, Tb1_0, Tb2_0, target0)
        b3_0 = 0

        # we will reduce B1 linearly to 0 over roll_days (like your example)
        if b1_0 % roll_days != 0:
            # still handle it by rounding daily decrement but keep integer
            b1_roll_per_day = b1_0 / roll_days
        else:
            b1_roll_per_day = b1_0 // roll_days

        # identify rollover window: last roll_days business days BEFORE Brent expiry
        # approximate by using Tb1 (time-to-expiry) == roll_days..1
        # (i.e., T-5 => Tb1=5, ..., T-1 => Tb1=1)
        for i in idx:
            Tb1 = int(df.loc[i, "Tb1"])
            Tb2 = int(df.loc[i, "Tb2"])
            Tb3 = int(df.loc[i, "Tb3"])
            Tw1 = int(df.loc[i, "Tw1"])
            Tw2 = int(df.loc[i, "Tw2"])

            # default: hold baseline
            B1 = b1_0
            W2 = 0

            # rollover window: Tb1 in [roll_days..1]
            if 1 <= Tb1 <= roll_days:
                step = roll_days - Tb1 + 1  # Tb1=5 -> step=1, ..., Tb1=1 -> step=5
                # WTI rolls
                W2 = min(total_wti, step * w_roll_per_day)
                W1 = total_wti - W2

                # Brent B1 decays to 0
                # use proportional decay from baseline b1_0
                # B1 at step k = round(b1_0 * (1 - k/roll_days))
                B1 = int(np.round(b1_0 * (1 - step / roll_days)))
                B1 = int(np.clip(B1, 0, total_brent))

                # match time exposure
                target = W1 * Tw1 + W2 * Tw2
                B2, B3 = solve_b2_b3_given_b1_integer(total_brent, B1, Tb1, Tb2, Tb3, target)

                df.loc[i, ["B1","B2","B3","W1","W2"]] = [B1, B2, B3, W1, W2]
            else:
                # hold baseline (W1 full)
                df.loc[i, ["B1","B2","B3","W1","W2"]] = [b1_0, b2_0, b3_0, total_wti, 0]

    return df.drop(columns=["month"])

# =========================================================
# TAB 2: WEIGHTED SPREAD TRADING
# =========================================================
with tab2:
    st.subheader("Weighted Spread Trading")

    # ---- Inputs: lots only ----
    cA, cB, cC = st.columns(3)
    with cA:
        total_brent = st.number_input("Total Brent lots", min_value=1, max_value=200, value=20, step=1)
    with cB:
        total_wti = st.number_input("Total WTI lots", min_value=1, max_value=200, value=20, step=1)
    with cC:
        roll_days = st.number_input("Rollover window", min_value=1, max_value=10, value=5, step=1)

    # ---- Load contract data ----
    df_c1 = load_data(tab_config["C1"])
    df_c2 = load_data(tab_config["C2"])
    df_c3 = load_data(tab_config["C3"])

    # ---- Merge all prices on Timestamp ----
    # Brent prices by tenor (C1,C2,C3)
    base = df_c1[["Timestamp"]].copy()

    base = base.merge(
        df_c1[["Timestamp","Brent_OPEN","Brent_HIGH","Brent_LOW","Brent_CLOSE","WTI_OPEN","WTI_HIGH","WTI_LOW","WTI_CLOSE"]],
        on="Timestamp", how="left"
    ).rename(columns={
        "Brent_OPEN":"Brent_OPEN_C1","Brent_HIGH":"Brent_HIGH_C1","Brent_LOW":"Brent_LOW_C1","Brent_CLOSE":"Brent_CLOSE_C1",
        "WTI_OPEN":"WTI_OPEN_C1","WTI_HIGH":"WTI_HIGH_C1","WTI_LOW":"WTI_LOW_C1","WTI_CLOSE":"WTI_CLOSE_C1",
    })

    base = base.merge(
        df_c2[["Timestamp","Brent_OPEN","Brent_HIGH","Brent_LOW","Brent_CLOSE","WTI_OPEN","WTI_HIGH","WTI_LOW","WTI_CLOSE"]],
        on="Timestamp", how="left"
    ).rename(columns={
        "Brent_OPEN":"Brent_OPEN_C2","Brent_HIGH":"Brent_HIGH_C2","Brent_LOW":"Brent_LOW_C2","Brent_CLOSE":"Brent_CLOSE_C2",
        "WTI_OPEN":"WTI_OPEN_C2","WTI_HIGH":"WTI_HIGH_C2","WTI_LOW":"WTI_LOW_C2","WTI_CLOSE":"WTI_CLOSE_C2",
    })

    base = base.merge(
        df_c3[["Timestamp","Brent_OPEN","Brent_HIGH","Brent_LOW","Brent_CLOSE"]],
        on="Timestamp", how="left"
    ).rename(columns={
        "Brent_OPEN":"Brent_OPEN_C3","Brent_HIGH":"Brent_HIGH_C3","Brent_LOW":"Brent_LOW_C3","Brent_CLOSE":"Brent_CLOSE_C3",
    })

    base["Timestamp"] = pd.to_datetime(base["Timestamp"]).dt.normalize()
    base = base.sort_values("Timestamp").reset_index(drop=True)

    # ---- Build lots schedule for ALL historical dates ----
    try:
        lots_df = build_lots_schedule(base["Timestamp"], total_brent=int(total_brent), total_wti=int(total_wti), roll_days=int(roll_days))
    except Exception as e:
        st.error(f"Lots schedule error: {e}")
        st.stop()

    # ---- Join lots onto prices ----
    dfw = base.merge(lots_df[["Timestamp","Tb1","Tb2","Tb3","Tw1","Tw2","B1","B2","B3","W1","W2"]], on="Timestamp", how="left")

    # ---- Compute weighted OHLC using lots as weights ----
    def wavg3(a1, a2, a3, w1, w2, w3):
        denom = (w1 + w2 + w3)
        return np.where(denom == 0, np.nan, (w1*a1 + w2*a2 + w3*a3) / denom)

    def wavg2(a1, a2, w1, w2):
        denom = (w1 + w2)
        return np.where(denom == 0, np.nan, (w1*a1 + w2*a2) / denom)

    # Brent weighted (B1,B2,B3)
    dfw["Brent_OPEN"]  = wavg3(dfw["Brent_OPEN_C1"],  dfw["Brent_OPEN_C2"],  dfw["Brent_OPEN_C3"],  dfw["B1"], dfw["B2"], dfw["B3"])
    dfw["Brent_HIGH"]  = wavg3(dfw["Brent_HIGH_C1"],  dfw["Brent_HIGH_C2"],  dfw["Brent_HIGH_C3"],  dfw["B1"], dfw["B2"], dfw["B3"])
    dfw["Brent_LOW"]   = wavg3(dfw["Brent_LOW_C1"],   dfw["Brent_LOW_C2"],   dfw["Brent_LOW_C3"],   dfw["B1"], dfw["B2"], dfw["B3"])
    dfw["Brent_CLOSE"] = wavg3(dfw["Brent_CLOSE_C1"], dfw["Brent_CLOSE_C2"], dfw["Brent_CLOSE_C3"], dfw["B1"], dfw["B2"], dfw["B3"])

    # WTI weighted (W1,W2)
    dfw["WTI_OPEN"]  = wavg2(dfw["WTI_OPEN_C1"],  dfw["WTI_OPEN_C2"],  dfw["W1"], dfw["W2"])
    dfw["WTI_HIGH"]  = wavg2(dfw["WTI_HIGH_C1"],  dfw["WTI_HIGH_C2"],  dfw["W1"], dfw["W2"])
    dfw["WTI_LOW"]   = wavg2(dfw["WTI_LOW_C1"],   dfw["WTI_LOW_C2"],   dfw["W1"], dfw["W2"])
    dfw["WTI_CLOSE"] = wavg2(dfw["WTI_CLOSE_C1"], dfw["WTI_CLOSE_C2"], dfw["W1"], dfw["W2"])

    df_base_weighted = dfw[["Timestamp","Brent_OPEN","Brent_HIGH","Brent_LOW","Brent_CLOSE",
                            "WTI_OPEN","WTI_HIGH","WTI_LOW","WTI_CLOSE"]].dropna().copy()

    # ---- Display lots + exposure check (latest date) ----
    last = dfw.dropna(subset=["B1","B2","B3","W1","W2"]).iloc[-1]
    br_expo = int(last["B1"]*last["Tb1"] + last["B2"]*last["Tb2"] + last["B3"]*last["Tb3"])
    wt_expo = int(last["W1"]*last["Tw1"] + last["W2"]*last["Tw2"])

    st.markdown(
        f"### Latest lots @ {pd.Timestamp(last['Timestamp']).strftime('%Y-%m-%d')}: "
        f"B1={int(last['B1'])}, B2={int(last['B2'])}, B3={int(last['B3'])} | "
        f"W1={int(last['W1'])}, W2={int(last['W2'])} "
        f"|| **Exposure: Brent={br_expo}, WTI={wt_expo}**"
    )

    with st.expander("Show lots schedule (last 60 rows)"):
        show_cols = ["Timestamp","Tb1","Tb2","Tb3","Tw1","Tw2","B1","B2","B3","W1","W2"]
        display_df = dfw[show_cols + ["Brent_CLOSE", "WTI_CLOSE"]].tail(60).copy()
        
        # Rename for clarity
        display_df = display_df.rename(columns={
            "Brent_CLOSE": "Brent_Weighted",
            "WTI_CLOSE": "WTI_Weighted"
        })
        
        # Style function with green highlight
        def highlight_weighted(s):
            if s.name in ['Brent_Weighted', 'WTI_Weighted']:
                return ['background-color: #90EE90; font-weight: bold; color: #000'] * len(s)
            return [''] * len(s)
        
        st.dataframe(
            display_df.style.apply(highlight_weighted).format({
                'Brent_Weighted': '{:.2f}',
                'WTI_Weighted': '{:.2f}'
            }),
            use_container_width=True
        )

    st.divider()
    render_spread_dashboard_streamlit(lookback, ma_window, visual_choice, show_ma, show_ma_sd, df_base_weighted)
    st.divider()
    render_ma_retracement_dashboard_streamlit(
        mr_lookback,
        int(mr_ma_window),
        visual_choice,
        MR_MIN_GAP_DAYS,
        MR_RETRACE_LEVELS,
        df_base=df_base_weighted
    )
    st.divider()
    render_persistence_dashboard_streamlit(p_lookback, fast_ma, slow_ma, df_base_weighted)
    
# =========================================================
# TAB 3: TRADING DAY ANALYSIS 
# =========================================================
with tab3:
    st.subheader("Trading Day Spread Distribution")

    # --- 1. CONFIGURATION ---
    FILE_PATH = "Brent_WTI_C1.xlsx"
    YEARS_TO_INCLUDE = st.number_input(
        "Lookback (Y)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="td_years_simple",
    )
    

    # --- 2. DATA PROCESSING ---
    @st.cache_data
    def load_and_process(path: str, years: int) -> pd.DataFrame:
        df = pd.read_excel(path)
        df.columns = [
            "Timestamp", "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE",
            "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE"
        ]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp")

        cutoff = df["Timestamp"].max() - pd.DateOffset(years=int(years))
        df = df[df["Timestamp"] >= cutoff].copy()

        df["spread_close"] = df["WTI_CLOSE"] - df["Brent_CLOSE"]
        df["Year"] = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["TradingDay"] = df.groupby(["Year", "Month"])["Timestamp"].rank(method="first").astype(int)

        return df

    df_plot = load_and_process(FILE_PATH, int(YEARS_TO_INCLUDE))

    # --- Current trading day and spread ---
    latest_date = df_plot["Timestamp"].max()
    current_trading_day = int(df_plot.loc[df_plot["Timestamp"] == latest_date, "TradingDay"].iloc[0])
    current_spread = float(df_plot.loc[df_plot["Timestamp"] == latest_date, "spread_close"].iloc[0])

    st.markdown(
        f"**Current Trading Day:** Day {current_trading_day} "
        f"({latest_date.strftime('%Y-%m-%d')}) | **Current Spread:** ${current_spread:.2f}"
    )


    # --- 3. CALCULATE STATISTICAL TABLE ---
    stats_table = (
        df_plot.groupby("TradingDay")["spread_close"]
        .agg(Mean="mean", Std_Dev="std", Skewness=skew)
        .reset_index()
    )

    stats_table["Mean"] = stats_table["Mean"].round(3)
    stats_table["Std_Dev"] = stats_table["Std_Dev"].round(3)
    stats_table["Skewness"] = stats_table["Skewness"].round(3)

    # --- 4a. OUTPUT TABLE ---
    st.markdown(f"### Distribution Stats (Last {int(YEARS_TO_INCLUDE)} Years)")
    st.dataframe(stats_table, use_container_width=True)

    # --- 4b. PLOT ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.set_theme(style="whitegrid")

    stats = df_plot.groupby("TradingDay")["spread_close"].describe()
    iqr = stats['75%'] - stats['25%']
    whisker_low = (stats['25%'] - 1.5 * iqr).min()
    whisker_high = (stats['75%'] + 1.5 * iqr).max()

    # Apply limits with a small 0.25 buffer to keep whiskers clear of the edge
    ax.set_ylim(whisker_low - 0.25, whisker_high + 0.25)

    # Main Boxplot
    sns.boxplot(
        x="TradingDay",
        y="spread_close",
        data=df_plot,
        palette="Blues_d",
        linewidth=1.2,
        fliersize=3,
        flierprops={"marker": "o", "markerfacecolor": "gray", "alpha": 0.2},
        ax=ax,
    )
    
    # Calculate statistics for each trading day
    stats_by_day = df_plot.groupby("TradingDay")["spread_close"].agg(['mean', 'std']).reset_index()
    stats_by_day['upper_95'] = stats_by_day['mean'] + 2 * stats_by_day['std']
    stats_by_day['lower_95'] = stats_by_day['mean'] - 2 * stats_by_day['std']

    # 95% confidence bands
    ax.fill_between(
        stats_by_day["TradingDay"] - 1,
        stats_by_day['lower_95'],
        stats_by_day['upper_95'],
        alpha=0.15,
        color='darkblue',
        label='95% Range (±2σ)',
        zorder=5
    )

    # Add lines for the boundaries
    ax.plot(stats_by_day["TradingDay"] - 1, stats_by_day['upper_95'], 
            linestyle='--', linewidth=1, color='blue', alpha=0.5, zorder=5)
    ax.plot(stats_by_day["TradingDay"] - 1, stats_by_day['lower_95'], 
            linestyle='--', linewidth=1, color='blue', alpha=0.5, zorder=5)
    
    # Set y-axis ticks to 0.25 bins
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    
    # Customizing the grid to look like your image
    ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.15)

    # Mean Line and Labeling
    ax.plot(
        stats_by_day["TradingDay"] - 1, 
        stats_by_day["mean"], 
        marker="o", linestyle="-", linewidth=2, markersize=4, 
        color="darkblue", label="Mean Spread", zorder=10
    )
    
    # --- HIGHLIGHT CURRENT DAY AND SPREAD ---
    # Add vertical line for current trading day
    ax.axvline(x=current_trading_day - 1, color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Current Day', zorder=8)
    
    # Add horizontal line for current spread
    ax.axhline(y=current_spread, color='orange', linestyle='--', 
               linewidth=2, alpha=0.7, label='Current Spread', zorder=8)
    
    # Add marker at intersection
    ax.scatter(current_trading_day - 1, current_spread, color='red', 
               s=150, marker='*', edgecolors='black', linewidth=1.5, 
               label='Current Position', zorder=11)
    
    # Calculate z-score for current position
    current_day_data = df_plot[df_plot["TradingDay"] == current_trading_day]["spread_close"]
    if len(current_day_data) > 1:
        day_mean = current_day_data.mean()
        day_std = current_day_data.std()
        z_score = (current_spread - day_mean) / day_std if day_std != 0 else 0
    else:
        day_mean = current_spread
        z_score = 0
    
    # Add annotation for current position
    y_frac = 0.85 if current_spread < day_mean else 0.15
    annotation_text = (
        f"Day {current_trading_day}\n"
        f"Spread: ${current_spread:.2f}\n"
        f"Z-score: {z_score:+.1f}σ"
    )
    ax.annotate(
        annotation_text,
        xy=(current_trading_day - 1, current_spread),
        xytext=(0.02, y_frac), textcoords="axes fraction",
        ha="left", va="center",
        fontsize=10, color="red", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="red", alpha=0.90, 
                  boxstyle="round,pad=0.5", linewidth=2),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        zorder=12
    )
    
    ax.set_title(
        f"Brent-WTI Spread Distribution by Trading Day (Last {int(YEARS_TO_INCLUDE)} Years)", 
        fontsize=16, fontweight="bold", pad=20
    )
    ax.set_ylabel("Spread $", fontsize=12)
    ax.set_xlabel("Trading Day of Month", fontsize=12)

    # Add legend
    ax.legend(loc='best', frameon=True, shadow=True)

    sns.despine(ax=ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# --- 5. CONDITIONAL PROBABILITY ANALYSIS ---
    st.markdown("---")
    st.markdown(f"### Conditional Probability: Next Day Forecast")
    
    # User inputs
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        input_day = st.number_input(
            "Current Trading Day",
            min_value=1,
            max_value=int(df_plot['TradingDay'].max()),
            value=current_trading_day,
            step=1,
            key="prob_day"
        )
    
    with col_input2:
        input_spread_lower = st.number_input(
            "Spread Range Lower Bound ($)",
            min_value=-10.0,
            max_value=5.0,
            value=float(np.floor(current_spread * 4) / 4),  # Round to nearest 0.25
            step=0.25,
            format="%.2f",
            key="prob_lower"
        )
    
    with col_input3:
        input_spread_upper = st.number_input(
            "Spread Range Upper Bound ($)",
            min_value=-10.0,
            max_value=5.0,
            value=float(np.floor(current_spread * 4) / 4) + 0.25,
            step=0.25,
            format="%.2f",
            key="prob_upper"
        )
    
    # Determine next trading day (handle month rollover)
    next_day = 1 if input_day >= 23 else input_day + 1
    
    st.markdown(
        f"**Query:** Given spread is in "
        f":green[**[${input_spread_lower:.2f}, ${input_spread_upper:.2f})**] "
        f"on **Day {input_day}**, what are the probabilities for **Day {next_day}**?"
    )
    
    if input_day >= 23:
        st.info("ℹ️ Day 23+ rolls over to Day 1 of the next month")
    
    # --- Show all historical spreads for the selected trading day ---
    st.markdown("---")
    st.markdown(f"#### Historical Spreads on Day {input_day}")
    
    # Get all instances of the selected trading day
    day_data = df_plot[df_plot['TradingDay'] == input_day].copy()
    day_data = day_data.sort_values('Timestamp', ascending=False)
    
    # Create display table
    day_display = day_data[['Timestamp', 'spread_close', 'Year', 'Month']].copy()
    day_display['Timestamp'] = pd.to_datetime(day_display['Timestamp']).dt.strftime('%Y-%m-%d')
    day_display.columns = ['Date', 'Spread', 'Year', 'Month']
    
    # Highlight rows that fall within the selected range
    def highlight_range(row):
        if input_spread_lower <= row['Spread'] < input_spread_upper:
            return ['background-color: #90EE90'] * len(row)  # Light green
        return [''] * len(row)
    
    in_range_count = len(day_display[(day_display['Spread'] >= input_spread_lower) & (day_display['Spread'] < input_spread_upper)])
    
    st.markdown(
        f"Found **{len(day_display)}** historical instances of Day {input_day}. "
        f"**{in_range_count}** fall within selected range "
        f"<span style='color: #00AA00; font-weight: bold;'>[${input_spread_lower:.2f}, ${input_spread_upper:.2f})</span> "
        f"(highlighted in <span style='background-color: #90EE90; padding: 2px 4px;'>green</span> below).",
        unsafe_allow_html=True
    )
    
    # Calculate statistics for this trading day
    col_day1, col_day2, col_day3, col_day4 = st.columns(4)
    col_day1.metric("Mean", f"${day_data['spread_close'].mean():.2f}")
    col_day2.metric("Median", f"${day_data['spread_close'].median():.2f}")
    col_day3.metric("Std Dev", f"${day_data['spread_close'].std():.2f}")
    col_day4.metric("Range",f"[\\${day_data['spread_close'].min():.2f}, \\${day_data['spread_close'].max():.2f}]")
    
    # Display table with styling
    styled_table = day_display.style.apply(highlight_range, axis=1)
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spread": st.column_config.NumberColumn(format="$%.2f"),
            "Year": st.column_config.NumberColumn(format="%d"),
            "Month": st.column_config.NumberColumn(format="%d")
        },
        height=400  # Scrollable if many rows
    )
    
    st.markdown("---")
    
    # Filter historical transitions from input day to next day
    transition_data = []
    
    for year in df_plot['Year'].unique():
        for month in df_plot['Month'].unique():
            month_data = df_plot[(df_plot['Year'] == year) & (df_plot['Month'] == month)].sort_values('TradingDay')
            
            # Get current day data
            current_day_data = month_data[month_data['TradingDay'] == input_day]
            
            if len(current_day_data) > 0:
                current_spread_hist = current_day_data['spread_close'].iloc[0]
                current_timestamp = current_day_data['Timestamp'].iloc[0]
                
                # Only proceed if current spread is in the specified range
                if input_spread_lower <= current_spread_hist < input_spread_upper:
                    # Handle next day (could be next month if Day 23+)
                    if input_day >= 23:
                        # Look for Day 1 of next month
                        next_month = month + 1 if month < 12 else 1
                        next_year = year if month < 12 else year + 1
                        next_month_data = df_plot[(df_plot['Year'] == next_year) & (df_plot['Month'] == next_month)].sort_values('TradingDay')
                        next_day_data = next_month_data[next_month_data['TradingDay'] == 1]
                    else:
                        # Same month, next trading day
                        next_day_data = month_data[month_data['TradingDay'] == next_day]
                    
                    if len(next_day_data) > 0:
                        next_spread_hist = next_day_data['spread_close'].iloc[0]
                        transition_data.append({
                            'current_spread': current_spread_hist,
                            'next_spread': next_spread_hist,
                            'year': year,
                            'month': month,
                            'date': current_timestamp
                        })
    
    if len(transition_data) == 0:
        st.warning(
            f"⚠️ No historical data found where Day {input_day} spread was in range "
            f"[${input_spread_lower:.2f}, ${input_spread_upper:.2f})"
        )
    else:
        transition_df = pd.DataFrame(transition_data)
        
        st.success(
            f"✓ Found **{len(transition_df)} historical instances** matching your criteria "
            f"(from {len(transition_df)} unique months)"
        )
        
        # Calculate next day outcome bins
        bin_size = 0.25
        next_day_spreads = transition_df['next_spread']
        
        min_spread = next_day_spreads.min()
        max_spread = next_day_spreads.max()
        
        bins = np.arange(
            np.floor(min_spread / bin_size) * bin_size,
            np.ceil(max_spread / bin_size) * bin_size + bin_size,
            bin_size
        )
        
        # Calculate probabilities
        counts, bin_edges = np.histogram(next_day_spreads, bins=bins)
        probabilities = (counts / counts.sum()) * 100
        
        # Create comprehensive probability table
        prob_table = pd.DataFrame({
            'Range_Lower': bin_edges[:-1],
            'Range_Upper': bin_edges[1:],
            'Count': counts,
            'Probability_%': probabilities
        })
        
        # Only show non-zero probabilities, sorted by probability
        prob_table = prob_table[prob_table['Count'] > 0].copy()
        prob_table = prob_table.sort_values('Probability_%', ascending=False).reset_index(drop=True)
        
        # Format the range column
        prob_table['Next_Day_Spread_Range'] = prob_table.apply(
            lambda row: f"[${row['Range_Lower']:.2f}, ${row['Range_Upper']:.2f})", axis=1
        )
        
        # Calculate cumulative probability
        prob_table['Cumulative_%'] = prob_table['Probability_%'].cumsum()
        
        # Final display table
        display_prob_table = prob_table[[
            'Next_Day_Spread_Range', 
            'Count', 
            'Probability_%', 
            'Cumulative_%'
        ]].copy()
        
        display_prob_table['Probability_%'] = display_prob_table['Probability_%'].round(2)
        display_prob_table['Cumulative_%'] = display_prob_table['Cumulative_%'].round(2)
        
        # Display
        st.markdown(f"#### Probability Table: Day {next_day} Outcomes")
        st.dataframe(
            display_prob_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Next_Day_Spread_Range": st.column_config.TextColumn("Next Day Spread Range"),
                "Count": st.column_config.NumberColumn("# Occurrences", format="%d"),
                "Probability_%": st.column_config.NumberColumn("Probability (%)", format="%.2f%%"),
                "Cumulative_%": st.column_config.NumberColumn("Cumulative (%)", format="%.2f%%")
            }
        )
        
        # Summary statistics
        st.markdown("---")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        mean_next = next_day_spreads.mean()
        median_next = next_day_spreads.median()
        std_next = next_day_spreads.std()
        most_likely_range = display_prob_table.iloc[0]['Next_Day_Spread_Range']
        
        col_stat1.metric("Mean Next Day", f"${mean_next:.2f}")
        col_stat2.metric("Median Next Day", f"${median_next:.2f}")
        col_stat3.metric("Std Dev", f"${std_next:.2f}")
        col_stat4.metric("Most Likely Range", most_likely_range)
        
        # Show the actual historical instances
        with st.expander(f"📋 View {len(transition_df)} Historical Instances"):
            history_display = transition_df[['date', 'current_spread', 'next_spread']].copy()
            history_display['date'] = pd.to_datetime(history_display['date']).dt.strftime('%Y-%m-%d')
            history_display['change'] = history_display['next_spread'] - history_display['current_spread']
            history_display.columns = ['Date', f'Day {input_day} Spread', f'Day {next_day} Spread', 'Change']
            history_display = history_display.sort_values('Date', ascending=False)
            
            st.dataframe(
                history_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    f'Day {input_day} Spread': st.column_config.NumberColumn(format="$%.2f"),
                    f'Day {next_day} Spread': st.column_config.NumberColumn(format="$%.2f"),
                    'Change': st.column_config.NumberColumn(format="$%.2f")
                }
            )
    
    # =========================================================
    # CURVATURE ANALYSIS
    # =========================================================
    st.markdown("---")
    st.subheader("Futures Curve Analysis")
    
    # --- Load multi-contract data using existing cached function ---
    @st.cache_data
    def load_curve_data(years: int) -> pd.DataFrame:
        # Reuse the existing load_data function from top of script
        df_c1 = load_data("Brent_WTI_C1.xlsx")
        df_c2 = load_data("Brent_WTI_C2.xlsx")
        df_c3 = load_data("Brent_WTI_C3.xlsx")
        
        # Apply lookback filter
        cutoff = df_c1["Timestamp"].max() - pd.DateOffset(years=years)
        df_c1 = df_c1[df_c1["Timestamp"] >= cutoff].copy()
        df_c2 = df_c2[df_c2["Timestamp"] >= cutoff].copy()
        df_c3 = df_c3[df_c3["Timestamp"] >= cutoff].copy()
        
        # Select only needed columns
        df_c1_sub = df_c1[["Timestamp", "Brent_CLOSE", "WTI_CLOSE"]]
        df_c2_sub = df_c2[["Timestamp", "Brent_CLOSE", "WTI_CLOSE"]]
        df_c3_sub = df_c3[["Timestamp", "Brent_CLOSE", "WTI_CLOSE"]]
        
        # Merge
        df = (
            df_c1_sub.merge(df_c2_sub, on="Timestamp", suffixes=("_C1", "_C2"))
                     .merge(df_c3_sub, on="Timestamp")
                     .rename(columns={"Brent_CLOSE": "Brent_CLOSE_C3", "WTI_CLOSE": "WTI_CLOSE_C3"})
        )
        
        need_cols = ["Brent_CLOSE_C1","Brent_CLOSE_C2","Brent_CLOSE_C3",
                     "WTI_CLOSE_C1","WTI_CLOSE_C2","WTI_CLOSE_C3"]
        df = df.dropna(subset=need_cols).reset_index(drop=True)
        
        return df
    
    df_curve = load_curve_data(int(YEARS_TO_INCLUDE))
    
    # --- Initialize session state for clear functionality ---
    if 'compare_key_counter' not in st.session_state:
        st.session_state.compare_key_counter = 0
    
    # --- Date selection controls ---
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        current_date_input = st.text_input(
            "Current Date (YYYY-MM-DD)",
            value=df_curve.iloc[-1]['Timestamp'].strftime('%Y-%m-%d'),
            key="curve_current_date_input",
            help="Enter date to view futures curve"
        )
    
    with col2:
        compare_date_input = st.text_input(
            "Compare Date (YYYY-MM-DD)",
            value="",
            key=f"curve_compare_date_input_{st.session_state.compare_key_counter}",
            help="Enter a date to overlay comparison curve"
        )
    
    with col3:
        if st.button("Clear", key="clear_curve_compare"):
            # Increment counter to create new widget with empty value
            st.session_state.compare_key_counter += 1
            st.rerun()
    
    # --- Extract current date data ---
    tenors = ["C1", "C2", "C3"]
    x_pos = np.arange(len(tenors))
    
    # Handle current date selection
    try:
        target_current = pd.to_datetime(current_date_input)
        current_idx = (df_curve['Timestamp'] - target_current).abs().idxmin()
        row_current = df_curve.iloc[current_idx]
        current_date = row_current['Timestamp']
    except:
        st.warning("Invalid current date format. Using latest date.")
        current_idx = len(df_curve) - 1
        row_current = df_curve.iloc[current_idx]
        current_date = row_current['Timestamp']
    
    brent_current = [row_current[f"Brent_CLOSE_{c}"] for c in tenors]
    wti_current = [row_current[f"WTI_CLOSE_{c}"] for c in tenors]
    spread_current = [w - b for w, b in zip(wti_current, brent_current)]
    
    # --- Handle comparison date ---
    show_comparison = False
    if compare_date_input.strip():
        try:
            target_date = pd.to_datetime(compare_date_input)
            compare_idx = (df_curve['Timestamp'] - target_date).abs().idxmin()
            row_compare = df_curve.iloc[compare_idx]
            compare_date = row_compare['Timestamp']
            
            brent_compare = [row_compare[f"Brent_CLOSE_{c}"] for c in tenors]
            wti_compare = [row_compare[f"WTI_CLOSE_{c}"] for c in tenors]
            spread_compare = [w - b for w, b in zip(wti_compare, brent_compare)]
            
            show_comparison = True
        except:
            st.warning("Invalid compare date format. Please use YYYY-MM-DD")
    
    # --- Display current date info ---
    df_curve_with_trading_day = df_curve.copy()
    df_curve_with_trading_day["Year"] = df_curve_with_trading_day["Timestamp"].dt.year
    df_curve_with_trading_day["Month"] = df_curve_with_trading_day["Timestamp"].dt.month
    df_curve_with_trading_day["TradingDay"] = df_curve_with_trading_day.groupby(
        ["Year", "Month"]
    )["Timestamp"].rank(method="first").astype(int)

    def get_td_str(df, target_date):
        match = df[df['Timestamp'] == target_date]
        if not match.empty:
            return f" (Trading Day {int(match['TradingDay'].iloc[0])})"
        return ""

    # --- Display current and compare date info with Trading Day ---
    st.markdown(f"**Current Date:** {current_date.strftime('%Y-%m-%d')}{get_td_str(df_curve_with_trading_day, current_date)}")

    if show_comparison:
        st.markdown(f"**Compare Date:** {compare_date.strftime('%Y-%m-%d')}{get_td_str(df_curve_with_trading_day, compare_date)}")
    
    # --- Create plots ---
    plt.close("all")
    
    with plt.style.context('dark_background'):
        fig_curve, (ax_futures, ax_spread_curve) = plt.subplots(
            2, 1, figsize=(12, 8), 
            gridspec_kw={'height_ratios': [2, 1]},
            facecolor='#0e1117'
        )
        
        # Set axes background color
        ax_futures.set_facecolor('#262730')
        ax_spread_curve.set_facecolor('#262730')
        
        # Plot 1: Futures Curve
        ax_futures.plot(x_pos, brent_current, marker="o", markersize=8, 
                       color='#00FFCC', label="Brent (Current)", linewidth=2)
        ax_futures.plot(x_pos, wti_current, marker="s", markersize=8, 
                       color='#FF3366', label="WTI (Current)", linewidth=2)
        
        if show_comparison:
            ax_futures.plot(x_pos, brent_compare, marker="o", markersize=6, 
                           color='#00FFCC', linewidth=1.5, linestyle='--', 
                           alpha=0.5, label="Brent (Compare)")
            ax_futures.plot(x_pos, wti_compare, marker="s", markersize=6, 
                           color='#FF3366', linewidth=1.5, linestyle='--', 
                           alpha=0.5, label="WTI (Compare)")
        
        ax_futures.set_xticks(x_pos)
        ax_futures.set_xticklabels(tenors, color='white')
        ax_futures.set_ylabel("Price ($/bbl)", fontsize=11, color='white')
        ax_futures.grid(True, alpha=0.15, linestyle='--', color='gray')
        ax_futures.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), 
                 frameon=True, shadow=False, facecolor='#1e1e1e', edgecolor='gray')
        ax_futures.tick_params(colors='white')
        ax_futures.spines['bottom'].set_color('gray')
        ax_futures.spines['top'].set_color('gray')
        ax_futures.spines['left'].set_color('gray')
        ax_futures.spines['right'].set_color('gray')
    
    # Plot 2: Spread Bars with Statistical Overlays
        bar_width = 0.35 if show_comparison else 0.5
        
        bars_current = ax_spread_curve.bar(
            x_pos - (bar_width/2 if show_comparison else 0), 
            spread_current, 
            width=bar_width,
            color=['#00FFCC' if s >= 0 else '#FF5500' for s in spread_current],
            alpha=0.7,
            label='Current Spread'
        )
        
        # Add value labels for current
        for i, (bar, val) in enumerate(zip(bars_current, spread_current)):
            ax_spread_curve.text(
                bar.get_x() + bar.get_width()/2, 
                val - 0.15 if val < 0 else val + 0.15,
                f'${val:.2f}',
                ha='center', 
                va='top' if val < 0 else 'bottom',
                fontsize=9,
                fontweight='bold',
                color='white'
            )
        
        if show_comparison:
            bars_compare = ax_spread_curve.bar(
                x_pos + bar_width/2, 
                spread_compare, 
                width=bar_width,
                color=['#888888' if s >= 0 else '#666666' for s in spread_compare],
                alpha=0.4,
                label='Compare Spread'
            )
            
            # Add value labels for comparison
            for i, (bar, val) in enumerate(zip(bars_compare, spread_compare)):
                ax_spread_curve.text(
                    bar.get_x() + bar.get_width()/2, 
                    val - 0.15 if val < 0 else val + 0.15,
                    f'${val:.2f}',
                    ha='center', 
                    va='top' if val < 0 else 'bottom',
                    fontsize=8,
                    color='#cccccc'
                )
        
        # --- ADD STATISTICAL OVERLAYS FOR ALL CONTRACTS ---
        # Calculate trading day for current date
        current_year = current_date.year
        current_month = current_date.month
        current_trading_day = int(df_plot[
            (df_plot['Timestamp'].dt.year == current_year) & 
            (df_plot['Timestamp'].dt.month == current_month) & 
            (df_plot['Timestamp'] == current_date)
        ]['TradingDay'].iloc[0]) if len(df_plot[
            (df_plot['Timestamp'].dt.year == current_year) & 
            (df_plot['Timestamp'].dt.month == current_month) & 
            (df_plot['Timestamp'] == current_date)
        ]) > 0 else None
        
        # Calculate spreads for all contracts
        df_curve_with_trading_day["spread_C1"] = (
            df_curve_with_trading_day["WTI_CLOSE_C1"] - df_curve_with_trading_day["Brent_CLOSE_C1"]
        )
        df_curve_with_trading_day["spread_C2"] = (
            df_curve_with_trading_day["WTI_CLOSE_C2"] - df_curve_with_trading_day["Brent_CLOSE_C2"]
        )
        df_curve_with_trading_day["spread_C3"] = (
            df_curve_with_trading_day["WTI_CLOSE_C3"] - df_curve_with_trading_day["Brent_CLOSE_C3"]
        )
        
        if current_trading_day is not None:
            # Loop through each contract and plot statistics
            for i, contract in enumerate(['C1', 'C2', 'C3']):
                day_data = df_curve_with_trading_day[
                    df_curve_with_trading_day['TradingDay'] == current_trading_day
                ][f'spread_{contract}']
                
                if len(day_data) > 0:
                    current_mean = day_data.mean()
                    current_std = day_data.std()
                    current_upper_95 = current_mean + 2 * current_std
                    current_lower_95 = current_mean - 2 * current_std
                    
                    # Plot mean and 95% range for current date
                    ax_spread_curve.errorbar(
                        x_pos[i] - (bar_width/2 if show_comparison else 0),
                        current_mean,
                        yerr=[[current_mean - current_lower_95], [current_upper_95 - current_mean]],
                        fmt='D',
                        color='#FFD700',
                        markersize=5,
                        capsize=6,
                        capthick=1.5,
                        elinewidth=1.5,   
                        alpha=0.7,
                        zorder=9
                    )
        
        # Add statistical overlay for compare date if present
        if show_comparison:
            compare_year = compare_date.year
            compare_month = compare_date.month
            compare_trading_day = int(df_plot[
                (df_plot['Timestamp'].dt.year == compare_year) & 
                (df_plot['Timestamp'].dt.month == compare_month) & 
                (df_plot['Timestamp'] == compare_date)
            ]['TradingDay'].iloc[0]) if len(df_plot[
                (df_plot['Timestamp'].dt.year == compare_year) & 
                (df_plot['Timestamp'].dt.month == compare_month) & 
                (df_plot['Timestamp'] == compare_date)
            ]) > 0 else None
            
            if compare_trading_day is not None:
                # Loop through each contract and plot statistics
                for i, contract in enumerate(['C1', 'C2', 'C3']):
                    compare_day_data = df_curve_with_trading_day[
                        df_curve_with_trading_day['TradingDay'] == compare_trading_day
                    ][f'spread_{contract}']
                    
                    if len(compare_day_data) > 0:
                        compare_mean = compare_day_data.mean()
                        compare_std = compare_day_data.std()
                        compare_upper_95 = compare_mean + 2 * compare_std
                        compare_lower_95 = compare_mean - 2 * compare_std
                        
                        # Plot mean and 95% range for compare date
                        ax_spread_curve.errorbar(
                            x_pos[i] + bar_width/2,
                            compare_mean,
                            yerr=[[compare_mean - compare_lower_95], [compare_upper_95 - compare_mean]],
                            fmt='D',
                            color='#FFD700',
                            markersize=5,
                            capsize=6,
                            capthick=1.5,
                            elinewidth=1.5,
                            alpha=0.7,
                            zorder=9
                        )
        
        ax_spread_curve.set_xticks(x_pos)
        ax_spread_curve.set_xticklabels(tenors, color='white')
        ax_spread_curve.set_ylabel("Spread", fontsize=11, color='white')
        ax_spread_curve.axhline(0, color='white', linewidth=1.2)
        ax_spread_curve.grid(True, alpha=0.15, axis='y', color='gray', linestyle='--')
        ax_spread_curve.tick_params(colors='white')
        ax_spread_curve.spines['bottom'].set_color('gray')
        ax_spread_curve.spines['top'].set_color('gray')
        ax_spread_curve.spines['left'].set_color('gray')
        ax_spread_curve.spines['right'].set_color('gray')

        # Auto-scale y-axis for spread with appropriate margins
        all_spreads = spread_current + (spread_compare if show_comparison else [])
        
        # Include statistical ranges in y-axis calculation
        if current_trading_day is not None:
            for contract in ['C1', 'C2', 'C3']:
                day_data = df_curve_with_trading_day[
                    df_curve_with_trading_day['TradingDay'] == current_trading_day
                ][f'spread_{contract}']
                if len(day_data) > 0:
                    all_spreads.extend([
                        day_data.mean() - 2 * day_data.std(),
                        day_data.mean() + 2 * day_data.std()
                    ])
        
        if show_comparison and compare_trading_day is not None:
            for contract in ['C1', 'C2', 'C3']:
                compare_day_data = df_curve_with_trading_day[
                    df_curve_with_trading_day['TradingDay'] == compare_trading_day
                ][f'spread_{contract}']
                if len(compare_day_data) > 0:
                    all_spreads.extend([
                        compare_day_data.mean() - 2 * compare_day_data.std(),
                        compare_day_data.mean() + 2 * compare_day_data.std()
                    ])
        
        s_min = min(all_spreads)
        s_max = max(all_spreads)
        margin = (s_max - s_min) * 0.15
        ax_spread_curve.set_ylim(s_min - margin, s_max + margin)
        
        fig_curve.tight_layout()
    st.pyplot(fig_curve, use_container_width=True)

