# Brent–WTI Spread Dashboards (Streamlit)

An interactive **dashboard** for analyzing **Brent–WTI crude oil spreads**, focusing on:
- Spread regimes and statistical positioning
- Moving-average trend diagnostics
- Persistence and decay of spread momentum

Designed for **market monitoring, trading context, and strategy diagnostics**.

---

## Features

### 1️⃣ Spread Dashboard
Visualizes different Brent–WTI spread constructions:
- **Close-to-Close**
- **High-to-High**
- **Floor** (WTI Low − Brent High)
- **Ceiling** (WTI High − Brent Low)

Includes:
- Rolling **Moving Average (MA)**
- **±1SD / ±2SD bands** around MA
- Global mean and SD reference lines
- **Z-score positioning**
- **Range persistence** (% of time spent in current SD bucket)
- Weekly & monthly **trend direction**

Key outputs:
- Current spread value
- SD regime
- Trend direction
- Historical persistence metrics

---

### 2️⃣ Trend Persistence Dashboard
Analyzes **momentum persistence** using MA differentials:
- Fast MA − Slow MA signal
- Peak / trough detection
- Signal decay analysis

Includes:
- Inflexion point detection
- Sensitivity analysis across multiple decay thresholds
- Average **time-to-decay**
- Average **$ move during decay**



