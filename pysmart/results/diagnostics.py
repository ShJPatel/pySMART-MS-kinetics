"""
diagnostics.py

Advanced diagnostic visualizers for SMART reaction pathway analysis:
- Gross inflow / outflow flux budget (2-panel stacked area)
- 1D & 2D phase portrait trajectories
- Top elementary producing / destroying reactions
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cantera as ct

def plot_gross_flux_budget(
    F: np.ndarray,
    time: np.ndarray,
    tracked_species: List[str],
    sub_mechs: List[str],
    palette: Dict[str, str],
    target_species: str = "NO",
    t_ms: bool = True,
    title: Optional[str] = None,
    ax_top: Optional[plt.Axes] = None,
    ax_bot: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plots a two-panel gross inflow (formation) and gross outflow (destruction)
    flux budget stacked area chart decomposing the target species turnover.
    """
    sp_idx = tracked_species.index(target_species)
    t_plot = time * 1e3 if t_ms else time
    nM = len(sub_mechs)
    nL = len(time)

    f_inflows = np.zeros((nM, nL))
    f_outflows = np.zeros((nM, nL))

    for k in range(nM):
        f_inflows[k] = np.sum(F[sp_idx, :, k, :], axis=0)
        f_outflows[k] = np.sum(F[:, sp_idx, k, :], axis=0)

    tot_in = f_inflows.sum(axis=0)
    tot_out = f_outflows.sum(axis=0)

    colors = [palette.get(m, "gray") for m in sub_mechs]

    if ax_top is None or ax_bot is None:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"hspace": 0.08})
    else:
        fig = ax_top.figure

    # Top panel: Gross inflow
    ax_top.stackplot(t_plot, f_inflows, labels=sub_mechs, colors=colors, alpha=0.8)
    ax_top.plot(t_plot, tot_in, color="black", linestyle="--", lw=1.2, label="Total Inflow")
    ax_top.set_ylabel(f"Gross Formation Flux\n[kmol/(m$^3\\cdot$s)]")
    ax_top.grid(True, linestyle=":", alpha=0.5)
    ax_top.legend(ncol=3, loc="upper right", frameon=True, fontsize=8)
    if title:
        ax_top.set_title(title)

    # Bottom panel: Gross outflow (inverted)
    ax_bot.stackplot(t_plot, f_outflows, labels=sub_mechs, colors=colors, alpha=0.8)
    ax_bot.plot(t_plot, tot_out, color="black", linestyle="--", lw=1.2, label="Total Outflow")
    ax_bot.invert_yaxis()
    ax_bot.set_ylabel(f"Gross Destruction Flux\n[kmol/(m$^3\\cdot$s)]")
    ax_bot.set_xlabel("Time [ms]" if t_ms else "Time [s]")
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    try:
        fig.tight_layout()
    except (UserWarning, ValueError):
        pass
    return fig, (ax_top, ax_bot)

def plot_phase_portrait(
    df: pd.DataFrame,
    U: np.ndarray,
    time: np.ndarray,
    tracked_species: List[str],
    sub_mechs: List[str],
    palette: Dict[str, str],
    target_species: str = "NO",
    x_col: str = "T",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots allocated source rates U_{i,k} against a state coordinate (e.g. Temperature T)."""
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    sp_idx = tracked_species.index(target_species)
    x_vals = df[x_col].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for k, m_name in enumerate(sub_mechs):
        u = U[sp_idx, k, :]
        if np.max(np.abs(u)) > 1e-12:
            ax.plot(x_vals, u, label=m_name, color=palette.get(m_name, "black"), lw=1.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Allocated Rate for {target_species} [kmol/(m$^3\\cdot$s)]")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    return fig, ax

def plot_top_reactions(
    df: pd.DataFrame,
    mech: str,
    sub_mech_dict: Dict[str, List[int]],
    target_species: str = "NO",
    time_col: str = "time",
    top_n: int = 8,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Ranks and visualizes top elementary producing (green) and destroying (red) reactions
    for the target species per sub-mechanism.
    """
    gas = ct.Solution(mech)
    n_reactions = gas.n_reactions
    t_val = df[time_col].values
    q_matrix = np.zeros((len(df), n_reactions))

    sp_names = gas.species_names
    for idx in range(len(df)):
        row = df.iloc[idx]
        T = row["T"]
        P = row.get("P", 101325.0)
        X = {sp: row.get(f"X_{sp}", row.get(sp, 0.0)) for sp in sp_names}
        gas.TPX = T, P, X
        q_matrix[idx, :] = gas.net_rates_of_progress

    # Integrate ROP contribution to target species
    integrated_q = np.array([np.trapz(q_matrix[:, r], x=t_val) for r in range(n_reactions)])
    rops = np.zeros(n_reactions)
    for r in range(n_reactions):
        rxn = gas.reaction(r)
        coeff = rxn.products.get(target_species, 0.0) - rxn.reactants.get(target_species, 0.0)
        rops[r] = coeff * integrated_q[r]

    # Group reactions by sub-mechanism
    plots_data = {}
    for mech_name, r_list in sub_mech_dict.items():
        unique_r = sorted(list(set(abs(int(r)) for r in r_list if 0 <= abs(int(r)) < n_reactions)))
        rxn_entries = []
        for r in unique_r:
            if abs(rops[r]) > 1e-12:
                rxn_entries.append({
                    "equation": gas.reaction(r).equation,
                    "rop": rops[r],
                    "abs_rop": abs(rops[r])
                })
        if rxn_entries:
            rxn_entries = sorted(rxn_entries, key=lambda x: x["abs_rop"], reverse=True)[:top_n]
            plots_data[mech_name] = rxn_entries

    if not plots_data:
        raise ValueError(f"No elementary reactions found producing or destroying {target_species}.")

    n_plots = len(plots_data)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (m_name, entries) in enumerate(plots_data.items()):
        ax = axes[idx]
        entries = entries[::-1]
        y_pos = np.arange(len(entries))
        rop_vals = [e["rop"] for e in entries]
        colors = ["#4CAF50" if r > 0 else "#F44336" for r in rop_vals]

        ax.barh(y_pos, rop_vals, color=colors, alpha=0.85, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([e["equation"] for e in entries], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(m_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Integrated Yield [kmol/m$^3$]")
        ax.grid(True, linestyle=":", alpha=0.5)

    for k in range(n_plots, len(axes)):
        fig.delaxes(axes[k])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
