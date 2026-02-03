import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import find_peaks

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Brent–WTI Dashboards", layout="wide")
st.title("Brent–WTI Dashboards")

# -----------------------------
# Tab Selection
# -----------------------------
tab1, tab2 = st.tabs(["Default", "Weighted"])

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_base = pd.read_excel(path)
    columns = [
        "Timestamp",
        "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE",
        "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE",
    ]
    df_base.columns = columns
    df_base["Timestamp"] = pd.to_datetime(df_base["Timestamp"])
    df_base = df_base.sort_values("Timestamp").reset_index(drop=True)
    return df_base

tab_config = {
    "C1": "Brent_WTI_C1.xlsx",
    "C2": "Brent_WTI_C2.xlsx",
    "C3": "Brent_WTI_C3.xlsx",
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
    check_interval = 1  # Check every 5 days
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

    # --- Sensitivity table ---
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
                if abs(float(a1)) <= float(lvl) * abs(float(a0)):
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
                if abs(float(a1)) <= float(lvl) * abs(float(a0_lat)):
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
    st.write("Retracement timing measures **how many days until |Amplitude| shrinks to X% of its starting value**.")
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
# TAB 2: WEIGHTED SPREAD TRADING
# =========================================================
with tab2:
    st.subheader("Weighted Spread Configuration")
    
    # Input fields for contract weights
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        brent_c1_weight = st.number_input("Brent C1 Contracts", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="brent_c1")
    with col2:
        brent_c2_weight = st.number_input("Brent C2 Contracts", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="brent_c2")
    with col3:
        brent_c3_weight = st.number_input("Brent C3 Contracts", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="brent_c3")
    with col4:
        wti_c1_weight = st.number_input("WTI C1 Contracts", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="wti_c1")
    with col5:
        wti_c2_weight = st.number_input("WTI C2 Contracts", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="wti_c2")
    
    # Load all contract data
    df_c1 = load_data(tab_config["C1"])
    df_c2 = load_data(tab_config["C2"])
    df_c3 = load_data(tab_config["C3"])
    
    # Calculate weighted prices
    def calculate_weighted_spread(df_c1, df_c2, df_c3, b1, b2, b3, w1, w2):
        """
        Calculate weighted spread: (WTI weighted avg) - (Brent weighted avg)
        """
        # Merge all dataframes on Timestamp
        df = df_c1[["Timestamp"]].copy()
        
        # Add Brent prices with contract suffixes
        df = df.merge(df_c1[["Timestamp", "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE"]], 
                      on="Timestamp", how="left", suffixes=("", "_C1"))
        df = df.merge(df_c2[["Timestamp", "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE"]], 
                      on="Timestamp", how="left", suffixes=("", "_C2"))
        df = df.merge(df_c3[["Timestamp", "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE"]], 
                      on="Timestamp", how="left", suffixes=("", "_C3"))
        
        # Rename C1 columns (they don't get suffix in first merge)
        df = df.rename(columns={
            "Brent_OPEN": "Brent_OPEN_C1",
            "Brent_HIGH": "Brent_HIGH_C1",
            "Brent_LOW": "Brent_LOW_C1",
            "Brent_CLOSE": "Brent_CLOSE_C1"
        })
        
        # Add WTI prices
        df = df.merge(df_c1[["Timestamp", "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE"]], 
                      on="Timestamp", how="left", suffixes=("", "_C1"))
        df = df.merge(df_c2[["Timestamp", "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE"]], 
                      on="Timestamp", how="left", suffixes=("", "_C2"))
        
        # Rename WTI C1 columns
        df = df.rename(columns={
            "WTI_OPEN": "WTI_OPEN_C1",
            "WTI_HIGH": "WTI_HIGH_C1",
            "WTI_LOW": "WTI_LOW_C1",
            "WTI_CLOSE": "WTI_CLOSE_C1"
        })
        
        # Calculate total weights
        total_brent = b1 + b2 + b3
        total_wti = w1 + w2
        
        if total_brent == 0 or total_wti == 0:
            st.error("Total Brent and WTI weights must be greater than 0")
            return None
        
        # Calculate weighted Brent prices
        df["Brent_OPEN"] = (b1 * df["Brent_OPEN_C1"] + b2 * df["Brent_OPEN_C2"] + b3 * df["Brent_OPEN_C3"]) / total_brent
        df["Brent_HIGH"] = (b1 * df["Brent_HIGH_C1"] + b2 * df["Brent_HIGH_C2"] + b3 * df["Brent_HIGH_C3"]) / total_brent
        df["Brent_LOW"] = (b1 * df["Brent_LOW_C1"] + b2 * df["Brent_LOW_C2"] + b3 * df["Brent_LOW_C3"]) / total_brent
        df["Brent_CLOSE"] = (b1 * df["Brent_CLOSE_C1"] + b2 * df["Brent_CLOSE_C2"] + b3 * df["Brent_CLOSE_C3"]) / total_brent
        
        # Calculate weighted WTI prices
        df["WTI_OPEN"] = (w1 * df["WTI_OPEN_C1"] + w2 * df["WTI_OPEN_C2"]) / total_wti
        df["WTI_HIGH"] = (w1 * df["WTI_HIGH_C1"] + w2 * df["WTI_HIGH_C2"]) / total_wti
        df["WTI_LOW"] = (w1 * df["WTI_LOW_C1"] + w2 * df["WTI_LOW_C2"]) / total_wti
        df["WTI_CLOSE"] = (w1 * df["WTI_CLOSE_C1"] + w2 * df["WTI_CLOSE_C2"]) / total_wti
        
        # Keep only the final columns
        df = df[["Timestamp", "Brent_OPEN", "Brent_HIGH", "Brent_LOW", "Brent_CLOSE",
                 "WTI_OPEN", "WTI_HIGH", "WTI_LOW", "WTI_CLOSE"]]
        
        return df
    
    df_base_weighted = calculate_weighted_spread(
        df_c1, df_c2, df_c3,
        brent_c1_weight, brent_c2_weight, brent_c3_weight,
        wti_c1_weight, wti_c2_weight
    )
    
    if df_base_weighted is not None:
        # Display weight summary
        st.info(f"**Weighted Spread Formula:** "
                f"[{wti_c1_weight:.1f}×WTI_C1 + {wti_c2_weight:.1f}×WTI_C2] / {wti_c1_weight + wti_c2_weight:.1f} - "
                f"[{brent_c1_weight:.1f}×Brent_C1 + {brent_c2_weight:.1f}×Brent_C2 + {brent_c3_weight:.1f}×Brent_C3] / {brent_c1_weight + brent_c2_weight + brent_c3_weight:.1f}")
        
        st.divider()
        
        # Render dashboards using the SAME unified sidebar controls
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
