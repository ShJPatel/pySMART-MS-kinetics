# AGENTS.md: Development & Architecture Rules for Autonomous Agents

This document defines architectural invariants, coding rules, and verification procedures for autonomous AI coding assistants (such as Antigravity, Claude, Copilot, or Cursor) working within the `pysmart` codebase.

---

## 1. System Invariants & Physical Constraints

1. **Exact Elemental Mass Conservation:**
   - Elemental conservation is non-negotiable. Across all time steps and stages:
     $$\sum_{k=1}^{N_{\mathrm{mech}}} u_{i,k,z} \equiv R_{i,z}, \qquad \sum_{k=1}^{N_{\mathrm{mech}}} Q_{i,k,z} \equiv P_{i,z}$$
   - In Flux Case 4 (depletion with a non-positive pool), if the global species production denominator is zero, you MUST fall back to the exact normalized initiation ratio:
     $$\nu_{i,k} \leftarrow \frac{N_{i,k}}{\sum_{l=1}^{N_{\mathrm{mech}}} N_{i,l}}$$
   - NEVER set unallocated rates or correction factors to zero, as this causes silent mass leakage.

2. **Decoupled Architecture:**
   - `pySMART` is a standalone library. Do NOT re-introduce dependencies on `AxiallyStagedCombustion`, `NRRL`, `ASC`, `ACC`, or thesis-specific analysis managers.
   - All external chemical kinetics operations must flow exclusively through standard `Cantera` APIs.

3. **Multi-Processing Boundaries:**
   - Cantera `Solution` objects (`ct.Solution`) cannot be pickled across process boundaries on Windows (`spawn`).
   - In multi-process workers (`parser.py` and `solver_multigrid.py`), pass only string mechanism paths and primitive data structures. Solutions must be instantiated inside the worker process using thread/process-local caches.

4. **NumBa JIT Constraints:**
   - Numerical kernels in `solver_serial.py` must run under `@numba.jit(nopython=True, fastmath=True)`.
   - Do NOT use Python dynamic structures (dictionaries, dynamic object attributes, variable-type lists) inside the JIT loops. Keep array layouts C-contiguous (`np.ascontiguousarray`).

---

## 2. Directory Layout & Module Responsibilities

- **`pysmart/api.py`**: High-level orchestrator (`smart.analyze`). Handles trajectory validation, cache routing, and automatic multi-grid vs. serial solver selection.
- **`pysmart/core/parser.py`**: Generates directed element flux matrices $G$ and initiation tensors $E$ from Cantera trajectories.
- **`pysmart/core/solver_serial.py`**: Base iterative solver and JIT time-stepping kernel.
- **`pysmart/core/solver_multigrid.py`**: 3-Level domain decomposition parallel solver (Level-0 serial coarse warmup $\to$ Level-1 parallel subdomain solves $\to$ Level-2 global restriction).
- **`pysmart/core/stage_tracker.py`**: Detects stage transitions and computes molar mixing jump vectors $\Delta P_i^{\mathrm{mix}}$.
- **`pysmart/mechanisms/`**: Mechanism classification, SubMechanism Enum, and curated mapping definitions.
- **`pysmart/results/container.py`**: `SMARTResult` object providing yield metrics, turnover timescales, and plotting interfaces.
- **`pysmart/results/network.py`**: NetworkX directed reaction graph visualizer.
- **`pysmart/results/diagnostics.py`**: Gross flux budgets, phase portraits, and top reaction bar charts.

---

## 3. Verification Protocol

Before submitting or committing any change, you MUST run the test suite and confirm 100% pass rate:

```bash
pytest tests/ -v
```

Ensure that:
1. `tests/test_basic.py` passes (single-stage conservation).
2. `tests/test_multistage.py` passes (SMART-MS jump tracking).
3. `tests/test_visualizers.py` passes (all 6 plotting routines render without exceptions).
