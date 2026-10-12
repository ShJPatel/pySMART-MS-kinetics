# pySMART: Sub-Mechanism Accounting and Reaction Tracing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

`pySMART` is a high-performance Python package for **reaction pathway analysis** in reacting flows and chemical kinetics. Building on the foundational SMART algorithm (Helbig & Bohon, 2024), `pySMART` introduces:

1. **Exact Open-System Mass Conservation (SMART-MS):** Resolves stage boundaries, discrete mixing jumps ($\Delta P_i^{\mathrm{mix}}$), and stream dilutions across multi-stage combustors with exact mass closure.
2. **Hierarchical 3-Level Multi-Grid Domain Decomposition (SMART-MG):** Eliminates serial trajectory bottlenecks across dense kinetic mechanisms (such as Glarborg 2018 with 83 nitrogen species), delivering up to **$109\times$ parallel speedup** without trajectory downsampling.
3. **Turnover Timescale Decomposition:** Evaluates path-resolved chemical turnover timescales ($\tau_{i,k}^{\mathrm{flux}}$) and relaxation timescales ($\tau_{i,k}^{\mathrm{flux,eq}}$).

---

## Installation

```bash
git clone https://github.com/spatel472/pysmart.git
cd pysmart
pip install -e .
```

---

## Quickstart

```python
import pandas as pd
import pysmart as smart

# 1. Provide your reacting trajectory as a DataFrame with T, P, X_species, time
# df can come from Cantera 0D reactor networks, 1D laminar flames, or CFD extractions
df = pd.read_csv("my_trajectory.csv")

# 2. Run reaction pathway analysis with a single call
result = smart.analyze(
    df=df,
    mech="gri30.yaml",
    time_col="time",
    tracked_element="N",
    target_species="NO",
    stage_col="stage"   # Optional: enables SMART-MS jump tracking across combustor stages
)

# 3. Inspect results
print(result.get_cumulative_yields())

# 4. Generate publication-ready figures
fig_rates, ax = result.plot_rates(target="NO")
fig_dom, ax = result.plot_dominance(target="NO")
fig_pools, ax = result.plot_pools(target="NO")
```

---

## Authors & Citation

Developed by Siddharth Patel.
Based on the SMART framework by H. Helbig and M. D. Bohon (Combustion and Flame, 2024).
