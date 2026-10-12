"""
api.py

Primary high-level API entry points for pySMART.
"""

from typing import Dict, List, Optional, Union
import os
import pickle
import numpy as np
import pandas as pd
import cantera as ct
import logging

from .core.parser import parse_cantera_data
from .core.stage_tracker import compute_mixing_jumps
from .core.solver_serial import run_smart_algorithm
from .core.solver_multigrid import run_smart_algorithm_multigrid
from .mechanisms.registry import get_mechanism_mapping
from .mechanisms.sub_mechanisms import SubMechanism
from .results.container import SMARTResult

logger = logging.getLogger(__name__)

def analyze(
    df: pd.DataFrame,
    mech: str,
    time_col: str = "time",
    tracked_element: str = "N",
    target_species: str = "NO",
    stage_col: Optional[str] = None,
    explicit_mixing_jumps: Optional[Dict] = None,
    mechanism_mapping: Optional[Union[Dict, str]] = None,
    multigrid: Optional[bool] = None,
    n_workers: Optional[int] = None,
    max_iter: int = 10,
    fallback_pressure_pa: Optional[float] = None,
    cache_path: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None
) -> SMARTResult:
    """
    Executes reaction pathway analysis (SMART-MS and SMART-MG) on a reacting flow trajectory.
    
    Args:
        df: pandas.DataFrame containing thermodynamic state (T, P, X_species) and time.
        mech: Cantera mechanism file path or name (e.g. 'Glarborg_2018.yaml', 'gri30.yaml').
        time_col: Column name representing time [s].
        tracked_element: Element to track ('N', 'C', etc.).
        target_species: Primary target species of interest ('NO', 'CO', etc.).
        stage_col: Optional column indicating combustor stage/zone for SMART-MS jump tracking.
        explicit_mixing_jumps: Optional dict overriding calculated mixing jumps across stages.
        mechanism_mapping: Optional custom sub-mechanism mapping dict or YAML path.
        multigrid: Whether to use SMART-MG domain decomposition. If None, auto-selects.
        n_workers: Parallel worker concurrency for parsing and multi-grid solves.
        max_iter: Maximum outer iterations.
        fallback_pressure_pa: Constant pressure in Pa if 'P' column is missing in df.
        cache_path: Optional path to save/load pre-parsed G and E flux matrices.
        palette: Optional color dictionary overriding standard sub-mechanism colors.
        
    Returns:
        SMARTResult: Container holding allocated pathway tensors, accessors, and plotters.
    """
    ct.suppress_thermo_warnings()
    gas = ct.Solution(mech)
    
    # 1. Resolve sub-mechanism mapping
    if mechanism_mapping is not None:
        if isinstance(mechanism_mapping, str):
            import yaml
            with open(mechanism_mapping, "r") as f:
                sub_mech_dict = yaml.safe_load(f)
        else:
            sub_mech_dict = mechanism_mapping
    else:
        sub_mech_dict = get_mechanism_mapping(mech, gas=gas)
        
    sub_mechs = list(sub_mech_dict.keys())
    k_mix = sub_mechs.index(SubMechanism.STAGED_MIXING.value) if SubMechanism.STAGED_MIXING.value in sub_mechs else -1
    
    # 2. Check cache for G and E matrices
    parsed_data = None
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading pre-parsed flux matrices from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            parsed_data = pickle.load(f)
            
    if parsed_data is None:
        logger.info(f"Parsing Cantera elemental flux graph for element '{tracked_element}'...")
        G, E, R, P, t, tracked_sp, parsed_sub_mechs = parse_cantera_data(
            df=df,
            mech_name=mech,
            time_col=time_col,
            tracked_element=tracked_element,
            sub_mech_dict=sub_mech_dict,
            n_workers=n_workers,
            fallback_pressure_pa=fallback_pressure_pa
        )
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((G, E, R, P, t, tracked_sp, parsed_sub_mechs), f)
    else:
        G, E, R, P, t, tracked_sp, parsed_sub_mechs = parsed_data

    # 3. Handle multi-stage mixing jumps (SMART-MS)
    delta_P_mix, stage_slices = compute_mixing_jumps(
        df=df,
        gas=gas,
        tracked_species=tracked_sp,
        stage_col=stage_col or "stage",
        fallback_pressure_pa=fallback_pressure_pa,
        explicit_jumps=explicit_mixing_jumps
    )
    
    # 4. Routing logic: Serial vs SMART-MG Multi-Grid
    nL = len(t)
    nS = len(tracked_sp)
    if multigrid is None:
        # Auto-route to SMART-MG if dense trajectory or large species set
        multigrid = bool((nL > 100 or nS > 25) and (n_workers != 1))
        
    if multigrid:
        logger.info(f"Solving using SMART-MG (Multi-Grid) parallel domain decomposition...")
        F, H, Q, U, N = run_smart_algorithm_multigrid(
            G=G, E=E, R=R, P=P, t=t,
            n_workers=n_workers,
            delta_P_mix=delta_P_mix,
            k_mix=k_mix,
            max_iter_level2=max_iter
        )
    else:
        logger.info(f"Solving using base SMART serial algorithm...")
        F, H, Q, U, N = run_smart_algorithm(
            G=G, E=E, R=R, P=P, t=t,
            max_iter=max_iter
        )
        
    return SMARTResult(
        F=F, H=H, Q=Q, U=U, N=N,
        time=t,
        tracked_species=tracked_sp,
        sub_mechs=sub_mechs,
        target_species=target_species,
        stage_slices=stage_slices,
        palette=palette
    )
