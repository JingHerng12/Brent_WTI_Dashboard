# Brent–WTI Spread Dashboards

An interactive dashboard for monitoring and diagnosing **Brent–WTI crude oil spreads**, built for:
- **Regime awareness** 
- **Trend + mean-reversion context** 
- **Persistence/decay** of momentum signals 

Designed for **market monitoring, trading context, and strategy diagnostics**.

---

## Features

### 1️⃣ Spread Dashboard
Visualizes multiple Brent–WTI spread constructions:
- **Close-to-Close**: `WTI_CLOSE − Brent_CLOSE`
- **High-to-High**: `WTI_HIGH − Brent_HIGH`
- **Floor**: `WTI_LOW − Brent_HIGH`
- **Ceiling**: `WTI_HIGH − Brent_LOW`

Includes:
- Rolling **Moving Average (MA)**
- Optional **MA bands**: **±1SD / ±2SD** (computed around MA)
- **Global mean** and **global SD** reference lines
- **SD bucket persistence**: % of time the spread stays in the current 1-SD z-score bucket
- Weekly & monthly **MA trend direction**

Key outputs:
- Current spread value and date
- Current **z-score** (vs global mean/SD)
- **Range frequency** (persistence in the current SD bucket)
- Weekly/monthly MA trend flags

---

### 2️⃣ MA Retracement Dashboard
Analyzes mean-reversion behavior around the moving average using:
- **Amplitude = Spread − MA**

Includes:
- **Peak/trough detection** on the amplitude series (turning points)
- **Retracement timing**: days until `|Amplitude|` shrinks to **X%** of its starting value  
  (e.g. 30% retrace = amplitude shrinks to 70% of start)
- **Sensitivity table** across multiple retracement levels
- “Latest start” panel with the most recent inflexion + projected retrace targets

Key outputs:
- MA window used
- # of detected inflexions
- Latest peak/trough start date + starting amplitude
- Average days-to-retrace + average $ move for each retrace threshold

---

### 3️⃣ Trend Persistence Dashboard
Measures persistence of spread momentum using an MA differential signal:
- **Signal = Fast MA − Slow MA**

Includes:
- Peak/trough detection on the **signal**
- **Decay timing**: days until `|Signal|` shrinks to **X%** of its starting value
- Sensitivity analysis across multiple decay thresholds
- “Latest signal start” panel with decay targets and dates

Key outputs:
- Fast/Slow MA settings
- # of signal inflexions
- Latest signal peak/trough start date + starting signal level
- Average time-to-decay + average $ move during decay

---

## Data Requirements

The Excel file must be named:
- `Brent_WTI_long_data_tidy.xlsx`

And contain columns in this order:
1. `Timestamp`
2. `Brent_OPEN`, `Brent_HIGH`, `Brent_LOW`, `Brent_CLOSE`
3. `WTI_OPEN`, `WTI_HIGH`, `WTI_LOW`, `WTI_CLOSE`

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
