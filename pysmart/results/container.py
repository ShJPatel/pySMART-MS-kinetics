"""
container.py

Structured result container for SMART pathway analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..mechanisms.sub_mechanisms import DEFAULT_PALETTE

class SMARTResult:
    """
    Container for reaction pathway analysis results.
    
    Attributes:
        F: (nS, nS, nM, nL) Allocated element flux tensor.
        H: (nS, nM, nL) Allocation factors.
        Q: (nS, nM, nL) Allocated running cumulative pool tensor.
        U: (nS, nM, nL) Allocated net source rate tensor.
        N: (nS, nM) Converged global pathway allocation ratios.
        time: (nL,) Time array.
        tracked_species: List of tracked species names.
        sub_mechs: List of sub-mechanism pathway names.
        stage_slices: Dict mapping stage names to trajectory slices.
    """
    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        U: np.ndarray,
        N: np.ndarray,
        time: np.ndarray,
        tracked_species: List[str],
        sub_mechs: List[str],
        target_species: str = "NO",
        stage_slices: Optional[Dict[str, slice]] = None,
        palette: Optional[Dict[str, str]] = None,
        iterations: Optional[int] = None
    ):
        self.F = F
        self.H = H
        self.Q = Q
        self.U = U
        self.N = N
        self.time = time
        self.tracked_species = tracked_species
        self.sub_mechs = sub_mechs
        self.target_species = target_species
        self.stage_slices = stage_slices or {"main": slice(0, len(time))}
        self.palette = {**DEFAULT_PALETTE, **(palette or {})}
        self.iterations = iterations
        
        self._sp_to_idx = {sp: i for i, sp in enumerate(tracked_species)}
        self._mech_to_idx = {m: i for i, m in enumerate(sub_mechs)}
        
    def _get_target_idx(self, target: Optional[str] = None) -> int:
        target_sp = target or self.target_species
        if target_sp not in self._sp_to_idx:
            raise ValueError(f"Species '{target_sp}' not in tracked species: {self.tracked_species}")
        return self._sp_to_idx[target_sp]

    def get_cumulative_yields(
        self,
        target: Optional[str] = None,
        t_span: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Integrates allocated source rates over time to compute cumulative formation
        and destruction yields for each sub-mechanism.
        
        Returns:
            pd.DataFrame: Columns ['sub_mechanism', 'formation_yield', 'destruction_yield', 'net_yield', 'formation_share_%']
        """
        sp_idx = self._get_target_idx(target)
        t = self.time
        mask = np.ones_like(t, dtype=bool)
        if t_span:
            mask = (t >= t_span[0]) & (t <= t_span[1])
            
        t_sub = t[mask]
        records = []
        tot_formation = 0.0
        
        for k_idx, m_name in enumerate(self.sub_mechs):
            u_traj = self.U[sp_idx, k_idx, mask]
            u_pos = np.maximum(u_traj, 0.0)
            u_neg = np.maximum(-u_traj, 0.0)
            
            y_pos = float(np.trapz(u_pos, t_sub))
            y_neg = float(np.trapz(u_neg, t_sub))
            y_net = y_pos - y_neg
            
            tot_formation += y_pos
            records.append({
                "sub_mechanism": m_name,
                "formation_yield": y_pos,
                "destruction_yield": y_neg,
                "net_yield": y_net
            })
            
        df_yields = pd.DataFrame(records)
        df_yields["formation_share_%"] = (df_yields["formation_yield"] / (tot_formation + 1e-30)) * 100.0
        return df_yields

    def summary(self, target: Optional[str] = None, t_span: Optional[Tuple[float, float]] = None) -> str:
        """
        Generates a formatted ASCII diagnostic summary table of cumulative yields,
        formation shares, and destruction percentages for terminal printing.
        """
        tgt = target or self.target_species
        df_y = self.get_cumulative_yields(target=tgt, t_span=t_span)
        
        tot_pos = df_y["formation_yield"].sum()
        tot_neg = df_y["destruction_yield"].sum()
        tot_net = df_y["net_yield"].sum()
        
        t_start = self.time[0] if t_span is None else t_span[0]
        t_end = self.time[-1] if t_span is None else t_span[1]
        
        lines = []
        lines.append("=" * 72)
        lines.append(f"  pySMART PATHWAY APPORTIONMENT SUMMARY: {tgt}")
        lines.append(f"  Domain: t = {t_start*1e3:.3f} ms to {t_end*1e3:.3f} ms (dt = {(t_end - t_start)*1e3:.3f} ms)")
        lines.append("=" * 72)
        lines.append(f"  {'Sub-Mechanism':<18} | {'Formation':>10} | {'Destruction':>11} | {'Net Yield':>10} | {'Share':>7}")
        lines.append(f"  {'-'*18}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}")
        
        for _, row in df_y.iterrows():
            m_name = row["sub_mechanism"]
            y_pos = row["formation_yield"]
            y_neg = row["destruction_yield"]
            y_net = row["net_yield"]
            share = row["formation_share_%"]
            
            # Highlight non-zero pathways
            if abs(y_pos) > 1e-15 or abs(y_neg) > 1e-15:
                lines.append(f"  {m_name:<18} | {y_pos:>10.2e} | {y_neg:>11.2e} | {y_net:>10.2e} | {share:>6.1f}%")
                
        lines.append(f"  {'-'*18}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}")
        lines.append(f"  {'GROSS FORMATION':<18} | {tot_pos:>10.2e} |             |            | 100.0%")
        lines.append(f"  {'GROSS DESTRUCTION':<18} |            | {tot_neg:>11.2e} |            |")
        lines.append(f"  {'NET SPECIES YIELD':<18} |            |             | {tot_net:>10.2e} | [kmol/m^3]")
        lines.append("=" * 72)
        return "\n".join(lines)

    def print_diagnostics(self, target: Optional[str] = None, t_span: Optional[Tuple[float, float]] = None) -> None:
        """
        Prints formatted pathway apportionment diagnostics directly to stdout.
        """
        print("\n" + self.summary(target=target, t_span=t_span) + "\n")

    def get_turnover_timescales(self, target: Optional[str] = None, target_conc_traj: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Computes path-specific turnover timescales tau_{i,k}^flux = [X_i] / |u_{i,k}|.
        """
        sp_idx = self._get_target_idx(target)
        timescales = {"time": self.time}
        for k_idx, m_name in enumerate(self.sub_mechs):
            u_mag = np.abs(self.U[sp_idx, k_idx, :])
            if target_conc_traj is not None:
                tau = np.where(u_mag > 1e-30, target_conc_traj / u_mag, np.inf)
            else:
                tau = np.where(u_mag > 1e-30, 1.0 / u_mag, np.inf)
            timescales[f"tau_{m_name}"] = tau
        return pd.DataFrame(timescales)

    def to_dataframe(self, target: Optional[str] = None) -> pd.DataFrame:
        """Exports time-resolved allocated pathway rates U for the target species to a DataFrame."""
        sp_idx = self._get_target_idx(target)
        data = {"time": self.time}
        for k_idx, m_name in enumerate(self.sub_mechs):
            data[f"U_{m_name}"] = self.U[sp_idx, k_idx, :]
            data[f"Q_{m_name}"] = self.Q[sp_idx, k_idx, :]
        return pd.DataFrame(data)

    def plot_rates(
        self,
        target: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show_net: bool = True,
        t_ms: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots time-resolved allocated pathway rates U_{i,k}."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            
        sp_idx = self._get_target_idx(target)
        t_plot = self.time * 1e3 if t_ms else self.time
        
        for k_idx, m_name in enumerate(self.sub_mechs):
            u = self.U[sp_idx, k_idx, :]
            if np.max(np.abs(u)) > 1e-12:
                c = self.palette.get(m_name, None)
                ax.plot(t_plot, u, label=m_name, color=c, lw=1.8)
                
        if show_net:
            u_net = np.sum(self.U[sp_idx, :, :], axis=0)
            ax.plot(t_plot, u_net, label="Net Rate", color="black", linestyle="--", lw=1.5)
            
        ax.set_xlabel("Time [ms]" if t_ms else "Time [s]")
        ax.set_ylabel(f"Allocated Rate for {target or self.target_species} [kmol/(m$^3\\cdot$s)]")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        return fig, ax

    def plot_dominance(
        self,
        target: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        t_ms: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots fractional formation share stackplot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            
        sp_idx = self._get_target_idx(target)
        t_plot = self.time * 1e3 if t_ms else self.time
        
        u_pos = np.maximum(self.U[sp_idx, :, :], 0.0)
        tot_pos = np.sum(u_pos, axis=0, keepdims=True)
        tot_pos = np.where(tot_pos > 1e-30, tot_pos, 1.0)
        shares = (u_pos / tot_pos) * 100.0
        
        # Filter active pathways
        active_idx = [k for k in range(len(self.sub_mechs)) if np.max(shares[k, :]) > 1.0]
        y_stack = [shares[k, :] for k in active_idx]
        labels = [self.sub_mechs[k] for k in active_idx]
        colors = [self.palette.get(self.sub_mechs[k], None) for k in active_idx]
        
        ax.stackplot(t_plot, y_stack, labels=labels, colors=colors, alpha=0.85)
        ax.set_xlabel("Time [ms]" if t_ms else "Time [s]")
        ax.set_ylabel("Formation Share [%]")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        return fig, ax

    def plot_pools(
        self,
        target: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        t_ms: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots cumulative sub-mechanism pools Q_{i,k}."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            
        sp_idx = self._get_target_idx(target)
        t_plot = self.time * 1e3 if t_ms else self.time
        
        for k_idx, m_name in enumerate(self.sub_mechs):
            q = self.Q[sp_idx, k_idx, :]
            if np.max(np.abs(q)) > 1e-12:
                c = self.palette.get(m_name, None)
                ax.plot(t_plot, q, label=f"Q: {m_name}", color=c, lw=1.8)
                
        q_tot = np.sum(self.Q[sp_idx, :, :], axis=0)
        ax.plot(t_plot, q_tot, label="Total Pool P", color="black", linestyle="--", lw=1.5)
        
        ax.set_xlabel("Time [ms]" if t_ms else "Time [s]")
        ax.set_ylabel(f"Cumulative Pool for {target or self.target_species} [kmol/m$^3$]")
        ax.legend(frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        return fig, ax

    def plot_network(
        self,
        df: Optional[pd.DataFrame] = None,
        t_span: Optional[Tuple[float, float]] = None,
        t_idx: Optional[int] = None,
        threshold: float = 0.05,
        target: Optional[str] = None,
        pos: Optional[Dict] = None,
        title: Optional[str] = "Reaction Pathway Network",
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes, Dict]:
        """
        Plots a directed element flux graph where nodes are species sized by concentration
        and edges are allocated elemental mass fluxes colored by sub-mechanism.
        """
        from .network import plot_reaction_network
        return plot_reaction_network(
            F=self.F,
            tracked_species=self.tracked_species,
            sub_mechs=self.sub_mechs,
            palette=self.palette,
            time=self.time,
            df=df,
            t_span=t_span,
            t_idx=t_idx,
            target_species=target or self.target_species,
            threshold=threshold,
            pos=pos,
            title=title,
            ax=ax
        )

    def plot_flux_budget(
        self,
        target: Optional[str] = None,
        t_ms: bool = True,
        title: Optional[str] = None,
        ax_top: Optional[plt.Axes] = None,
        ax_bot: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plots a 2-panel gross inflow and outflow flux budget stacked area chart."""
        from .diagnostics import plot_gross_flux_budget
        return plot_gross_flux_budget(
            F=self.F,
            time=self.time,
            tracked_species=self.tracked_species,
            sub_mechs=self.sub_mechs,
            palette=self.palette,
            target_species=target or self.target_species,
            t_ms=t_ms,
            title=title,
            ax_top=ax_top,
            ax_bot=ax_bot
        )

    def plot_phase_portrait(
        self,
        df: pd.DataFrame,
        x_col: str = "T",
        target: Optional[str] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots allocated source rates U_{i,k} against a state coordinate (e.g. Temperature T)."""
        from .diagnostics import plot_phase_portrait
        return plot_phase_portrait(
            df=df,
            U=self.U,
            time=self.time,
            tracked_species=self.tracked_species,
            sub_mechs=self.sub_mechs,
            palette=self.palette,
            target_species=target or self.target_species,
            x_col=x_col,
            title=title,
            ax=ax
        )

