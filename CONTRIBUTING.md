# Contributing to pySMART

Thank you for your interest in contributing to **pySMART**! Whether you are a human combustion researcher, software engineer, or an autonomous AI coding agent, this guide outlines our development standards, codebase architecture, and testing protocols.

---

## 1. Development Environment Setup

### Prerequisites
- Python >= 3.9 (recommended: 3.10 or 3.11)
- Git
- Conda or virtualenv

### Setting Up a Development Clone
```bash
git clone https://github.com/ShJPatel/pysmart.git
cd pysmart

# Create and activate an environment (example using conda)
conda create -n pysmart-dev python=3.10 -y
conda activate pysmart-dev

# Install dependencies including dev tools
pip install -e ".[dev]"
```

---

## 2. Codebase Architecture

`pySMART` is organized into clean, modular layers:

```
pysmart/
├── api.py                  # High-level user interface (smart.analyze)
├── core/
│   ├── parser.py           # Cantera flux graph pre-processor (evaluates G and E matrices)
│   ├── solver_serial.py    # JIT-compiled base SMART solver (NumBa-accelerated inner loop)
│   ├── solver_multigrid.py # Hierarchical 3-level domain decomposition (SMART-MG)
│   └── stage_tracker.py    # Multi-stage boundary jump detection & pool dilution (SMART-MS)
├── mechanisms/
│   ├── sub_mechanisms.py   # SubMechanism Enum & standard color palette definitions
│   ├── registry.py         # Curated mapping loader & heuristic gateway auto-classifier
│   └── glarborg_2018_v2.json # Curated 11-pathway mapping for Glarborg 2018
└── results/
    ├── container.py        # SMARTResult container (yield accessors, export helpers)
    ├── diagnostics.py      # Gross flux budget & state-space phase portrait plots
    └── network.py          # Directed element flux network graph visualizer (NetworkX)
```

---

## 3. Core Design Principles & Coding Standards

1. **Exact Elemental Mass Conservation:**
   - Any algorithm modification must preserve elemental closure ($\sum_k u_{i,k} \equiv R_i$ and $\sum_k Q_{i,k} \equiv P_i$) across all grid states.
   - Never set unallocated fluxes to zero in Case 4. Always use the normalized fallback ratio ($N_{i,k} / \sum_l N_{i,l}$).

2. **NumPy & NumBa Performance:**
   - Numerical kernels in `solver_serial.py` are JIT-compiled with `@numba.jit(nopython=True)`.
   - Avoid Python object allocations (dicts, lists, strings) inside the inner time-stepping loops (`run_smart_timestepping_jit`). Keep array accesses contiguous where possible.

3. **Multi-Processing & Parallel Safety:**
   - `solver_multigrid.py` relies on `ProcessPoolExecutor`.
   - All worker arguments and returns must be standard picklable Python primitives or NumPy arrays. Never pass `ct.Solution` instances across process boundaries; instantiate solutions inside worker processes or use in-memory caches.

4. **Matplotlib Visualizer Contract:**
   - Plotting methods must accept an optional `ax=None` parameter.
   - When `ax` is passed, plot on that axis; when omitted, instantiate a new `plt.subplots()`.
   - Always return the `(fig, ax)` tuple (or `(fig, ax, pos)` for network graphs) to allow downstream modification or saving.

---

## 4. How to Add a Curated Mechanism Mapping

If you want to add support for a new chemical kinetic mechanism (e.g., Okafor, San Diego, CRECK):

1. **Define Sub-Mechanism Reaction Gateway Indices:**
   - Identify which elementary reactions represent the primary unfixing/initiation steps for each sub-mechanism (e.g., thermal dissociation, hydrocarbon attack on $\ch{N2}$, fuel-bound amine attack).
2. **Create the Mapping JSON:**
   - Format 1-based signed indices (positive for forward reaction as written in mechanism, negative for reverse) and save under `pysmart/mechanisms/<mech_name>_mapping.json`.
3. **Register in `registry.py`:**
   - Add a loader hook in `pysmart/mechanisms/registry.py` to route to your curated JSON.

---

## 5. Testing & Verification Guidelines

All pull requests must pass the existing test suite without regressions.

### Running Tests Locally
```bash
pytest -v
```

### Writing New Tests
- Place tests in the `tests/` directory prefixed with `test_`.
- Unit tests should use compact reactor trajectories ($\le 100$ states) so the test suite finishes in under 15 seconds.
- Assert both tensor shape consistency and numerical conservation:
  ```python
  # Ensure allocated sources sum to the net production rate
  u_sum = np.sum(result.U[target_idx, :, :], axis=0)
  assert np.allclose(u_sum, expected_net_rate, atol=1e-12)
  ```

---

## 6. Pull Request & Commit Guidelines

1. **Commit Messages:**
   - Use clear, descriptive commit messages (e.g., `feat: Add Okafor 2018 mechanism mapping`, `fix: Preserve pool sign in Case 3 drawdown`).
2. **Branching Strategy:**
   - Create a feature branch off `main`: `git checkout -b feature/my-feature`.
   - Submit a Pull Request targeting `main`.
