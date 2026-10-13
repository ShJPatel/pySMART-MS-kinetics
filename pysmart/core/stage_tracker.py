"""
stage_tracker.py

Implements SMART-MS multi-stage mass jump evaluation and boundary alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cantera as ct

def detect_stage_transitions(df: pd.DataFrame, stage_col: str = "stage") -> List[int]:
    """
    Identifies the integer row indices where combustor stages transition.
    
    A transition index z corresponds to the first row of stage s+1.
    """
    if stage_col not in df.columns:
        return []
    stage_series = df[stage_col].values
    transitions = []
    for i in range(1, len(stage_series)):
        if stage_series[i] != stage_series[i - 1]:
            transitions.append(i)
    return transitions

def compute_mixing_jumps(
    df: pd.DataFrame,
    gas: ct.Solution,
    tracked_species: List[str],
    stage_col: str = "stage",
    fallback_pressure_pa: Optional[float] = None,
    explicit_jumps: Optional[Dict[int, Dict[str, float]]] = None,
    verbose: int = 1
) -> Tuple[np.ndarray, Dict[str, slice]]:
    """
    Evaluates the discrete molar jump vector Delta P_i^mix = [X_i]_entry - [X_i]_exit
    for each tracked species across stage transitions.
    
    Returns:
        delta_P_mix: (nS, nL) array of stage mixing jumps.
        stage_slices: dict mapping stage identifier to slice(start, end).
    """
    nS = len(tracked_species)
    nL = len(df)
    delta_P_mix = np.zeros((nS, nL), dtype=np.float64)
    
    if stage_col not in df.columns:
        if verbose >= 2:
            print(f"[pySMART:StageTracker] Single stage trajectory detected ({nL:,} points).")
        return delta_P_mix, {"single_stage": slice(0, nL)}
        
    transitions = detect_stage_transitions(df, stage_col=stage_col)
    
    # Compute stage boundaries
    stage_slices = {}
    unique_stages = df[stage_col].unique()
    for st in unique_stages:
        st_indices = np.where(df[stage_col] == st)[0]
        stage_slices[str(st)] = slice(int(st_indices[0]), int(st_indices[-1]) + 1)
        
    if not transitions:
        if verbose >= 2:
            print(f"[pySMART:StageTracker] Column '{stage_col}' found but no stage transitions detected.")
        return delta_P_mix, stage_slices

    if verbose >= 1:
        print(f"[pySMART:StageTracker] Multi-stage combustor detected ({len(unique_stages)} stages, {len(transitions)} transition boundaries):")
        for st_name, sl in stage_slices.items():
            print(f"  * Stage '{st_name}': rows {sl.start:,} to {sl.stop:,} ({sl.stop - sl.start:,} steps)")

    for t_idx in transitions:
        st_prev = df[stage_col].iloc[t_idx - 1]
        st_next = df[stage_col].iloc[t_idx]
        
        # Check if explicit override is provided
        if explicit_jumps and st_next in explicit_jumps:
            override = explicit_jumps[st_next]
            for i, sp in enumerate(tracked_species):
                if sp in override:
                    delta_P_mix[i, t_idx] = override[sp]
            if verbose >= 2:
                print(f"    -> Boundary row {t_idx:,} ('{st_prev}' -> '{st_next}'): Applied explicit jump override.")
            continue
            
        # Default Auto-detection from adjacent rows
        row_exit = df.iloc[t_idx - 1]
        row_entry = df.iloc[t_idx]
        
        T_exit = row_exit["T"]
        P_exit = row_exit.get("P", fallback_pressure_pa or 101325.0)
        T_entry = row_entry["T"]
        P_entry = row_entry.get("P", fallback_pressure_pa or 101325.0)
        
        # Build X vectors
        X_exit = {sp: row_exit.get(f"X_{sp}", row_exit.get(sp, 0.0)) for sp in gas.species_names}
        X_entry = {sp: row_entry.get(f"X_{sp}", row_entry.get(sp, 0.0)) for sp in gas.species_names}
        
        gas.TPX = T_exit, P_exit, X_exit
        rho_exit = gas.density
        Y_exit = gas.Y
        
        gas.TPX = T_entry, P_entry, X_entry
        rho_entry = gas.density
        Y_entry = gas.Y
        
        top_jumps = []
        for i, sp in enumerate(tracked_species):
            sp_idx = gas.species_index(sp)
            W_i = gas.molecular_weights[sp_idx]
            # [X_i] in kmol/m^3 = (rho * Y_i) / W_i
            c_entry = (rho_entry * Y_entry[sp_idx]) / W_i
            c_exit = (rho_exit * Y_exit[sp_idx]) / W_i
            jump = c_entry - c_exit
            delta_P_mix[i, t_idx] = jump
            if abs(jump) > 1e-6:
                top_jumps.append((sp, jump))
                
        if verbose >= 2:
            top_jumps.sort(key=lambda x: abs(x[1]), reverse=True)
            jump_str = ", ".join([f"{sp}: {j:+.2e}" for sp, j in top_jumps[:4]])
            print(f"    -> Boundary row {t_idx:,} ('{st_prev}' -> '{st_next}'): T={T_exit:.0f}K -> {T_entry:.0f}K | delta_P_mix: [{jump_str}] kmol/m^3")
            
    return delta_P_mix, stage_slices
