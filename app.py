import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Brent–WTI Dashboards", layout="wide")
st.title("Brent–WTI Dashboards")

# -----------------------------
# Load Data  
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_base = pd.read_excel(path)
    columns = ['Timestamp', 'Brent_OPEN', 'Brent_HIGH', 'Brent_LOW', 'Brent_CLOSE',
               'WTI_OPEN', 'WTI_HIGH', 'WTI_LOW', 'WTI_CLOSE']
    df_base.columns = columns
    df_base['Timestamp'] = pd.to_datetime(df_base['Timestamp'])
    df_base = df_base.sort_values('Timestamp').reset_index(drop=True)
    return df_base

DATA_PATH = "Brent_WTI_long_data_tidy.xlsx"
df_base = load_data(DATA_PATH)

# =========================================================
# Sidebar controls 
# =========================================================
st.sidebar.header("Spread Dashboard Controls")
lookback = st.sidebar.number_input("Lookback (Y)", min_value=1, max_value=30, value=3, step=1)
ma_window = st.sidebar.number_input("MA Window (D)", min_value=5, max_value=400, value=50, step=5)

view_map = {
    "Close": "spread_close",
    "High": "spread_high",
    "Floor": "spread_WTI_low_Brent_high",
    "Ceiling": "spread_WTI_high_Brent_low",
}
visual_label = st.sidebar.selectbox("View:", options=list(view_map.keys()), index=0)
visual_choice = view_map[visual_label]

show_ma = st.sidebar.checkbox("Show MA", value=True)
show_ma_sd = st.sidebar.checkbox("Show MA Bands", value=True)

st.sidebar.divider()
st.sidebar.header("Persistence Dashboard Controls")
p_lookback = st.sidebar.number_input("Persistence Lookback (Y)", min_value=1, max_value=30, value=3, step=1)
fast_ma = st.sidebar.number_input("Fast MA Window", min_value=2, max_value=300, value=20, step=1)
slow_ma = st.sidebar.number_input("Slow MA Window", min_value=5, max_value=600, value=50, step=1)

# =========================================================
# 1) Spread Dashboard
# =========================================================
def render_spread_dashboard_streamlit(lookback, ma_window, visual_choice, show_ma, show_ma_sd):
    # Filter Data
    latest_date = df_base['Timestamp'].max()
    cutoff_date = latest_date - pd.DateOffset(years=int(lookback))
    df = df_base[df_base['Timestamp'] >= cutoff_date].copy()

    # Calculate Spreads
    df['spread_close'] = df['WTI_CLOSE'] - df['Brent_CLOSE']
    df['spread_high'] = df['WTI_HIGH'] - df['Brent_HIGH']
    df['spread_WTI_low_Brent_high'] = df['WTI_LOW'] - df['Brent_HIGH']
    df['spread_WTI_high_Brent_low'] = df['WTI_HIGH'] - df['Brent_LOW']

    labels_map = {
        'spread_close': ('Close-to-Close', '#2E86C1'),
        'spread_high': ('High-to-High', '#27AE60'),
        'spread_WTI_low_Brent_high': ('Floor (WTI Low - Brent High)', '#E74C3C'),
        'spread_WTI_high_Brent_low': ('Ceiling (WTI High - Brent Low)', '#8E44AD')
    }
    label, color = labels_map[visual_choice]
    regime_series = df[visual_choice]

    # --- Setup Figure ---
    plt.close('all')
    fig = plt.figure(figsize=(15, 8))

    # Plot main series
    plt.plot(df['Timestamp'], regime_series, label=label, color=color, lw=1.8, alpha=0.9, zorder=4)

    # --- Trend Logic & Moving Average ---
    trend_w, trend_m = "N/A", "N/A"
    ma_stats_text = ""

    if show_ma:
        safe_window = min(int(ma_window), len(regime_series))
        ma_line = regime_series.rolling(window=safe_window).mean()
        plt.plot(df['Timestamp'], ma_line, label=f'{safe_window}D MA', color='black', lw=1.2, alpha=0.8, zorder=3)

        curr_sma = ma_line.iloc[-1]
        week_sma = ma_line.iloc[-6] if len(ma_line) > 6 else np.nan
        month_sma = ma_line.iloc[-22] if len(ma_line) > 22 else np.nan
        trend_w = "UP" if curr_sma > week_sma else "DOWN"
        trend_m = "UP" if curr_sma > month_sma else "DOWN"

        if show_ma_sd:
            ma_sd = regime_series.rolling(window=safe_window).std()
            u1, u2 = ma_line + ma_sd, ma_line + 2*ma_sd
            l1, l2 = ma_line - ma_sd, ma_line - 2*ma_sd

            # Prepare text for the "NOW" annotation
            ma_stats_text = (f"\nMA ±1SD: {l1.iloc[-1]:.2f} / {u1.iloc[-1]:.2f}"
                             f"\nMA ±2SD: {l2.iloc[-1]:.2f} / {u2.iloc[-1]:.2f}")

            plt.fill_between(df['Timestamp'], l1, u1, color='#1ABC9C', alpha=0.15, label='MA ±1SD', zorder=2)
            plt.fill_between(df['Timestamp'], l2, l1, color='#9B59B6', alpha=0.1, label='MA ±2SD', zorder=1)
            plt.fill_between(df['Timestamp'], u1, u2, color='#9B59B6', alpha=0.1, zorder=1)

    # Global Statistics
    ref_mean, ref_std = regime_series.mean(), regime_series.std()
    current_val, current_date = regime_series.iloc[-1], df['Timestamp'].iloc[-1]
    z_score = (current_val - ref_mean) / ref_std

    # Range Persistence
    lower_sd_limit = np.floor(z_score)
    upper_sd_limit = lower_sd_limit + 1
    in_range_mask = regime_series.apply(lambda x: lower_sd_limit <= (x - ref_mean)/ref_std < upper_sd_limit)
    persistence_pct = (in_range_mask.sum() / len(regime_series)) * 100

    # Reference Lines (Global)
    plt.axhline(ref_mean, color='black', lw=2, alpha=0.4, label='Global Mean')
    plt.axhline(ref_mean + ref_std, color='gray', lw=1, ls='--', alpha=0.5)
    plt.axhline(ref_mean - ref_std, color='gray', lw=1, ls='--', alpha=0.5)
    plt.axhline(ref_mean + 2*ref_std, color='#C0392B', lw=1.5, ls=':', alpha=0.7)
    plt.axhline(ref_mean - 2*ref_std, color='#C0392B', lw=1.5, ls=':', alpha=0.7)

    # --- Annotations & Right Axis Labels ---
    curr_xlim = plt.xlim()
    plt.xlim(curr_xlim[0], curr_xlim[1] + (curr_xlim[1] - curr_xlim[0]) * 0.22)
    text_x = plt.xlim()[1]

    # Scatter & Comprehensive "NOW" Annotation
    plt.scatter(current_date, current_val, color='blue', s=60, zorder=6)
    plt.annotate(f'Current Spread: {current_val:.2f} ({z_score:+.1f} SD)'
                 f'\nRange Freq: {persistence_pct:.1f}%'
                 f'{ma_stats_text}',
                 xy=(current_date, current_val), xytext=(text_x, current_val),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 va='center', ha='right', fontsize=9, color='blue', fontweight='bold')

    # GLOBAL SD LABELS (RIGHT SIDE)
    plt.text(text_x, ref_mean, f' Mean: {ref_mean:.2f}', va='bottom', ha='right', fontsize=8, alpha=0.7)
    plt.text(text_x, ref_mean + ref_std, f' +1SD: {ref_mean + ref_std:.2f}', va='bottom', ha='right', fontsize=7, color='gray')
    plt.text(text_x, ref_mean - ref_std, f' -1SD: {ref_mean - ref_std:.2f}', va='top', ha='right', fontsize=7, color='gray')
    plt.text(text_x, ref_mean + 2 * ref_std, f' +2SD: {ref_mean + 2*ref_std:.2f}', va='bottom', ha='right', fontsize=7, color='#C0392B', fontweight='bold')
    plt.text(text_x, ref_mean - 2 * ref_std, f' -2SD: {ref_mean - 2*ref_std:.2f}', va='top', ha='right', fontsize=7, color='#C0392B', fontweight='bold')

    # Status Bar
    plt.text(0.5, 0.98, f"SMA Trend: [Weekly: {trend_w}] | [Monthly: {trend_m}]",
             transform=plt.gca().transAxes, ha='center', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'),
             fontsize=10, fontweight='bold')

    # Formatting & Legend
    plt.title(f'Heatmap Dashboard | {int(lookback)}Y History | {label}', loc='left', fontsize=12, pad=25)
    plt.legend(loc='upper left', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.1)

    # Bucket Table 
    dev = (regime_series - ref_mean).dropna()
    bucket = (np.round(dev / 0.5) * 0.5)

    counts = pd.Series(bucket).value_counts().sort_index(ascending=False)

    half_width = 0.25  # because bucket size is 0.5, so +/- 0.25 around the center

    table_data = []
    for center, v in counts[counts > 0].items():
        lo = center - half_width
        hi = center + half_width
        # Range label like: [+0.75, +1.25)
        range_label = f"[{lo:+.2f}, {hi:+.2f})"
        table_data.append([range_label, int(v)])

    table = plt.table(
        cellText=table_data,
        colLabels=['$ Dev Range', 'Count'],
        loc='upper left',
        bbox=[1.02, 0.2, 0.3, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)   

    plt.tight_layout()

    # Streamlit display (instead of plt.show + print)
    st.subheader("1) Spread Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current spread", f"{current_val:.2f}")
    c2.metric("Range Freq", f"{persistence_pct:.1f}%")
    c3.metric("Weekly trend (MA)", trend_w)
    c4.metric("Monthly trend (MA)", trend_m)

    st.pyplot(fig, use_container_width=True)

    st.markdown("### Analysis Snapshot")
    st.write(f"View: **{label}**")
    st.write(f"{persistence_pct:.1f}% of time spent in {lower_sd_limit:.0f}SD to {upper_sd_limit:.0f}SD bucket.")
    st.write(f"SMA Trend (Lookback {int(ma_window)}D): Weekly={trend_w} | Monthly={trend_m}")

# =========================================================
# 2) Persistence Dashboard 
# =========================================================
def render_persistence_dashboard_streamlit(lookback, fast_ma, slow_ma):
    # Filter Data
    latest_date = df_base['Timestamp'].max()
    cutoff_date = latest_date - pd.DateOffset(years=lookback)
    df = df_base[df_base['Timestamp'] >= cutoff_date].copy()
    
    # Calculate Spread (Close-to-Close)
    s = df['WTI_CLOSE'] - df['Brent_CLOSE']
    s.index = df['Timestamp']
    s = s.sort_index().dropna()

    # Signal Construction
    if fast_ma >= slow_ma:
        print("Error: Fast MA must be smaller than Slow MA.")
        return

    ma_fast = s.rolling(fast_ma).mean()
    ma_slow = s.rolling(slow_ma).mean()
    signal = ma_fast - ma_slow

    # Inflexion Detection logic
    slope = ma_fast.diff(5)
    turn = np.sign(slope).diff()
    inflexions = []
    last_t = None
    for t in turn.index:
        if pd.isna(turn.loc[t]) or pd.isna(signal.loc[t]): continue
        if turn.loc[t] == -2: typ = "peak"
        elif turn.loc[t] == 2: typ = "trough"
        else: continue
        if last_t is not None and (t - last_t).days < 20: continue
        if abs(signal.loc[t]) < 0.2: continue
        inflexions.append((t, typ))
        last_t = t

    # --- Sensitivity Analysis Logic (Global Stats) ---
    # This populates the summary_rows variable to fix the NameError
    decay_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    summary_rows = []
    for d_frac in decay_values:
        wane_days, wane_diffs = [], []
        for t0, typ in inflexions:
            s0 = signal.loc[t0]
            if pd.isna(s0) or s0 == 0: continue
            future = signal.loc[t0:].iloc[1:250]
            for d, (t, val) in enumerate(future.items(), start=1):
                if abs(val) <= d_frac * abs(s0):
                    wane_days.append(d)
                    wane_diffs.append(abs(s.loc[t] - s.loc[t0]))
                    break
        avg_days = np.mean(wane_days) if wane_days else 0
        std_days = np.std(wane_days) if wane_days else 0
        avg_move = np.mean(wane_diffs) if wane_diffs else 0
        summary_rows.append([f"{(1 - d_frac)*100:.0f}%", f"{avg_days:.1f}", f"±{std_days:.1f}", f"${avg_move:.2f}"])

    # --- Plotting Setup ---
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top Panel: Price + MAs + Legend
    ax1.plot(s.index, s.values, label="Actual Spread", color='lightgray', alpha=0.5, zorder=1)
    ax1.plot(ma_fast.index, ma_fast.values, label=f"Fast {fast_ma}D", color='#2E86C1', lw=2, zorder=3)
    ax1.plot(ma_slow.index, ma_slow.values, label=f"Slow {slow_ma}D", color='#E67E22', lw=2, zorder=2)
    ax1.set_title(f"Spread Trend Persistence Dashboard | {lookback}Y History", fontsize=14)
    ax1.set_ylabel("Price ($/BBL)")
    ax1.legend(loc='upper left', fontsize=9, frameon=True)

    # Bottom Panel: Momentum Signal + Triangles
    ax2.plot(signal.index, signal.values, label="Momentum Signal", color='#27AE60', lw=1.5)
    ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
    
    peak_dates = [t for t, typ in inflexions if typ == 'peak']
    trough_dates = [t for t, typ in inflexions if typ == 'trough']
    
    if peak_dates: 
        ax2.scatter(peak_dates, signal.loc[peak_dates], marker="v", color='red', s=100, label="Peak", zorder=5)
    if trough_dates: 
        ax2.scatter(trough_dates, signal.loc[trough_dates], marker="^", color='blue', s=100, label="Trough", zorder=5)
    
    ax2.set_ylabel("Signal Amplitude")
    ax2.legend(loc='upper left', fontsize=9)

    # --- INFO TABLES ---
    if inflexions:
        t_lat, typ_lat = inflexions[-1]
        s0_lat = signal.loc[t_lat]
        
        # Base Data for Latest Inflexion
        inflex_data = [
            ["Type", typ_lat.upper()],
            ["Date", t_lat.strftime('%Y-%m-%d')],
            ["Start Price", f"${s.loc[t_lat]:.2f}"]
        ]
        
        # Calculate specific retrace prices for THIS signal
        target_levels = [0.70, 0.50, 0.30, 0.10] # 30%, 50%, 70%, 90%
        future_lat = signal.loc[t_lat:].iloc[1:]
        
        for lvl in target_levels:
            retrace_pct = int((1 - lvl) * 100)
            target_display = "Pending"
            for t_f, val_f in future_lat.items():
                if abs(val_f) <= lvl * abs(s0_lat):
                    target_display = f"${s.loc[t_f]:.2f} ({t_f.strftime('%Y-%m-%d')})"
                    break
            inflex_data.append([f"{retrace_pct}% Level", target_display])

        # Render Table 1: Latest Info
        inflex_ax = fig.add_axes([1.02, 0.60, 0.40, 0.25]) 
        inflex_ax.axis('off')
        inflex_ax.set_title("LATEST SIGNAL & RETRACE PRICES", fontsize=11, fontweight='bold', color='darkred' if typ_lat == 'peak' else 'darkblue')
        t1 = inflex_ax.table(cellText=inflex_data, loc='center', cellLoc='left')
        t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1.2, 1.8)

    # Render Table 2: Historical Sensitivity
    table_ax = fig.add_axes([1.02, 0.15, 0.40, 0.35])
    table_ax.axis('off')
    table_ax.set_title("Historical Retrace Sensitivity", fontsize=11, fontweight='bold', pad=20)
    tbl = table_ax.table(cellText=summary_rows, colLabels=['Retrace %', 'Avg Days', '1 SD', 'Avg $ Move'], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.5)

    plt.show()

    # Streamlit display
    c1, c2, c3 = st.columns(3)
    c1.metric("Fast MA", int(fast_ma))
    c2.metric("Slow MA", int(slow_ma))
    c3.metric("# Inflexions", len(inflexions))

    st.pyplot(fig, use_container_width=True)

# =========================================================
# Render both dashboards
# =========================================================
render_spread_dashboard_streamlit(
    lookback=lookback,
    ma_window=ma_window,
    visual_choice=visual_choice,
    show_ma=show_ma,
    show_ma_sd=show_ma_sd
)

st.divider()

render_persistence_dashboard_streamlit(
    lookback=p_lookback,
    fast_ma=fast_ma,
    slow_ma=slow_ma
)
