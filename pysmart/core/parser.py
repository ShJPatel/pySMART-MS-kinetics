import numpy as np
import pandas as pd
import cantera as ct
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_REACTIONS_CACHE = {}

def _add_reaction_flux(flux_matrix, gas, tracked_element, rate, reactants, products):
    """Helper to add flux from a single reaction to the flux matrix."""
    if rate == 0:
        return

    # Calculate number of tracked atoms in reactants and products for this reaction
    r_atoms = {gas.species_index(sp): coeff * gas.n_atoms(sp, tracked_element)
               for sp, coeff in reactants.items() if gas.n_atoms(sp, tracked_element) > 0}
    p_atoms = {gas.species_index(sp): coeff * gas.n_atoms(sp, tracked_element)
               for sp, coeff in products.items() if gas.n_atoms(sp, tracked_element) > 0}

    total_r_atoms = sum(r_atoms.values())

    # Element balance check
    if total_r_atoms == 0 or sum(p_atoms.values()) == 0:
        return

    # Proportional distribution of elemental fluxes from reactants to products
    for r_idx, r_val in r_atoms.items():
        for p_idx, p_val in p_atoms.items():
            if r_idx == p_idx:
                continue
            transfer_atoms = (r_val * p_val) / total_r_atoms
            flux = rate * transfer_atoms
            flux_matrix[p_idx, r_idx] += flux

def get_element_fluxes(gas, net_rates, tracked_element):
    """
    Computes the elemental fluxes between all species for a given state.
    
    Args:
        gas: cantera.Solution object at a specific state.
        net_rates: array of net reaction rates of progress.
        tracked_element: str, the element to track (e.g., 'N').
        
    Returns:
        flux_matrix: 2D array of shape (n_species, n_species)
                     flux_matrix[i, j] is the elemental flux from species j to species i.
    """
    n_sp = gas.n_species
    n_rxn = gas.n_reactions
    flux_matrix = np.zeros((n_sp, n_sp))
    
    # Check if element exists in mechanism
    if gas.element_index(tracked_element) < 0:
        raise ValueError(f"Element {tracked_element} not found in mechanism.")
    
    global _REACTIONS_CACHE
    gas_id = id(gas)
    if gas_id not in _REACTIONS_CACHE:
        _REACTIONS_CACHE[gas_id] = gas.reactions()
    rxns = _REACTIONS_CACHE[gas_id]
    
    for r in range(n_rxn):
        q = net_rates[r]
        if q == 0:
            continue
            
        rxn = rxns[r]
        
        # Determine direction
        if q > 0:
            reactants = rxn.reactants
            products = rxn.products
        else:
            reactants = rxn.products
            products = rxn.reactants
            
        _add_reaction_flux(flux_matrix, gas, tracked_element, abs(q), reactants, products)

    # Convert to directed graph where g_{i,j} = -g_{j,i} and negative values are eliminated
    for i in range(n_sp):
        for j in range(n_sp):
            if i == j:
                continue
            net_flux = flux_matrix[i, j] - flux_matrix[j, i]
            if net_flux > 0:
                flux_matrix[i, j] = net_flux
                flux_matrix[j, i] = 0.0
            elif net_flux < 0:
                flux_matrix[i, j] = 0.0
                flux_matrix[j, i] = -net_flux
            else:
                flux_matrix[i, j] = 0.0
                flux_matrix[j, i] = 0.0

    return flux_matrix

_GAS_SOLUTIONS = {}

def _process_single_state(args):
    """
    Worker function to process a single thermodynamic state in parallel.
    We must instantiate a Cantera Solution object inside the worker because
    ct.Solution objects cannot be pickled/sent across processes easily.
    """
    idx, row_dict, mech_name, tracked_element, sub_mech_dict, tracked_species_indices = args
    
    global _GAS_SOLUTIONS
    if mech_name not in _GAS_SOLUTIONS:
        ct.suppress_thermo_warnings()
        _GAS_SOLUTIONS[mech_name] = ct.Solution(mech_name)
    gas = _GAS_SOLUTIONS[mech_name]
    nS = len(tracked_species_indices)
    nM = len(sub_mech_dict)
    sub_mechs = list(sub_mech_dict.keys())
    
    T = row_dict.get('T', None)
    if T is None:
        raise KeyError(
            "'T' (temperature) not found in row dict. "
            "Ensure the DataFrame contains a 'T' column."
        )
    if 'P' not in row_dict:
        raise KeyError(
            "'P' (pressure in Pa) not found in row dict. "
            "Pass fallback_pressure_pa=<value_in_Pa> to parse_cantera_data(), "
            "or add a 'P' column to your DataFrame."
        )
    P_val = row_dict['P']
    # Extract species fractions
    X = np.zeros(gas.n_species)
    sp_names = gas.species_names
    
    has_X = False
    for i, sp in enumerate(sp_names):
        if sp in row_dict:
            X[i] = row_dict[sp]
            has_X = True
        elif f"X_{sp}" in row_dict:
            X[i] = row_dict[f"X_{sp}"]
            has_X = True
            
    if not has_X:
        Y = np.zeros(gas.n_species)
        for i, sp in enumerate(sp_names):
            if f"Y_{sp}" in row_dict:
                Y[i] = row_dict[f"Y_{sp}"]
        if np.sum(Y) > 0:
            gas.TPY = T, P_val, Y
        else:
            gas.TPX = T, P_val, X
    else:
        gas.TPX = T, P_val, X
        
    net_rates = gas.net_rates_of_progress
    
    # Matrices for this specific Z step
    G_z = np.zeros((nS, nS))
    E_z = np.zeros((nS, nS, nM))
    
    # 1. Global flux matrix G_z
    flux_matrix = get_element_fluxes(gas, net_rates, tracked_element)
    
    # Extract only tracked species
    for i_idx, i_global in enumerate(tracked_species_indices):
        for j_idx, j_global in enumerate(tracked_species_indices):
            G_z[i_idx, j_idx] = flux_matrix[i_global, j_global]
            
    # 2. Initiation flux matrix E_z
    global _REACTIONS_CACHE
    gas_id = id(gas)
    if gas_id not in _REACTIONS_CACHE:
        _REACTIONS_CACHE[gas_id] = gas.reactions()
    rxns = _REACTIONS_CACHE[gas_id]
    
    for k_idx, mech in enumerate(sub_mechs):
        rxn_indices = sub_mech_dict[mech]
        sub_flux_matrix = np.zeros((gas.n_species, gas.n_species))
        
        for r_idx_signed in rxn_indices:
            is_reverse_only = (r_idx_signed < 0) or (isinstance(r_idx_signed, float) and str(r_idx_signed) == '-0.0')
            r_idx = int(abs(r_idx_signed))
            rxn = rxns[r_idx]
            q = net_rates[r_idx]
            
            if q == 0:
                continue
                
            if not is_reverse_only and q > 0:
                _add_reaction_flux(sub_flux_matrix, gas, tracked_element, q, rxn.reactants, rxn.products)
            elif is_reverse_only and q < 0:
                _add_reaction_flux(sub_flux_matrix, gas, tracked_element, abs(q), rxn.products, rxn.reactants)

        # Convert to directed graph for initiation fluxes as well
        for i in range(gas.n_species):
            for j in range(gas.n_species):
                if i == j:
                    continue
                net_flux = sub_flux_matrix[i, j] - sub_flux_matrix[j, i]
                if net_flux > 0:
                    sub_flux_matrix[i, j] = net_flux
                    sub_flux_matrix[j, i] = 0.0
                elif net_flux < 0:
                    sub_flux_matrix[i, j] = 0.0
                    sub_flux_matrix[j, i] = -net_flux
                else:
                    sub_flux_matrix[i, j] = 0.0
                    sub_flux_matrix[j, i] = 0.0

        for i_idx, i_global in enumerate(tracked_species_indices):
            for j_idx, j_global in enumerate(tracked_species_indices):
                E_z[i_idx, j_idx, k_idx] = sub_flux_matrix[i_global, j_global]
                
    return idx, G_z, E_z

def parse_cantera_data(
    df,
    mech_name: str,
    time_col: str,
    tracked_element: str,
    sub_mech_dict: dict,
    n_workers: int = None,
    fallback_pressure_pa: float = None,
    mixing_jumps: dict = None,
    verbose: int = 1,
):
    """
    Parses a user-specified dataframe to generate G, E, R, P matrices.
    Parallelized using concurrent.futures.ProcessPoolExecutor.

    Args:
        df: pandas.DataFrame containing the thermodynamic state and species
            mass/mole fractions.
        mech_name (str): Name of the Cantera mechanism file (e.g., 'gri30.yaml').
        time_col (str): Name of the column in df that represents time or distance.
        tracked_element (str): The element to track (e.g., 'N').
        sub_mech_dict (dict): Maps sub-mechanism names to lists of reaction indices
            (0-based).
        n_workers (int, optional): Number of parallel processes. Defaults to
            ``os.cpu_count() - 2``.
        fallback_pressure_pa (float, optional): Constant pressure [Pa] to use when
            the DataFrame does **not** contain a ``'P'`` column.  Required if the
            combustor runs at a fixed pressure that is not stored per row.  A
            ``ValueError`` is raised immediately if ``'P'`` is absent and this
            argument is omitted, preventing silent wrong-pressure evaluations.

    Returns:
        G, E, R_mat, P, t, tracked_species_names, sub_mechs
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()-2

    is_daemon = multiprocessing.current_process().daemon

    if is_daemon:
        logger.info("Running in a daemon process, forcing sequential execution to avoid child process restrictions...")
        n_workers = 0

    # Dummy gas to get static metadata
    gas_dummy = ct.Solution(mech_name)
    tracked_species_indices = [i for i, sp in enumerate(gas_dummy.species_names) 
                               if gas_dummy.n_atoms(sp, tracked_element) > 0]
                               
    nS = len(tracked_species_indices)
    nL = len(df)
    nM = len(sub_mech_dict)
    sub_mechs = list(sub_mech_dict.keys())
    
    t = df[time_col].values

    if verbose >= 1:
        sp_sample = ', '.join([gas_dummy.species_names[i] for i in tracked_species_indices[:8]])
        if nS > 8:
            sp_sample += "..."
        print(f"[pySMART:Parser] Parsing Cantera elemental flux graph:")
        print(f"  * Element: '{tracked_element}' | Trajectory states: {nL:,} | Sub-mechanisms: {nM}")
        print(f"  * Tracked species ({nS}): {sp_sample}")

    # Resolve pressure: prefer a per-row 'P' column; require an explicit fallback
    # if it is missing so we never silently evaluate at the wrong pressure.
    has_pressure_col = 'P' in df.columns
    if not has_pressure_col:
        if fallback_pressure_pa is None:
            raise ValueError(
                "DataFrame has no 'P' column and fallback_pressure_pa was not "
                "provided. Pass fallback_pressure_pa=<value_in_Pa> (e.g. "
                "c.P_inlet or P_atm * ct.one_atm) to parse_cantera_data()."
            )
        if verbose >= 2:
            print(f"  * Fallback pressure: {fallback_pressure_pa:.0f} Pa ({fallback_pressure_pa / ct.one_atm:.2f} atm)")
        logger.info(
            f"No 'P' column in DataFrame using prescribed 'fallback'  pressure "
            f"{fallback_pressure_pa:.0f} Pa ({fallback_pressure_pa / ct.one_atm:.2f} atm) "
            f"into all row dicts."
        )

    # Prepare arguments for the process pool.
    # We convert df rows to dicts because pandas Series objects are slow to pickle.
    df_dicts = df.to_dict('records')
    if not has_pressure_col:
        for d in df_dicts:
            d['P'] = fallback_pressure_pa
    worker_args = [(idx, row_dict, mech_name, tracked_element, sub_mech_dict, tracked_species_indices)
                   for idx, row_dict in enumerate(df_dicts)]
    # Execute processing
    G = np.zeros((nS, nS, nL))
    E = np.zeros((nS, nS, nM, nL))
    
    completed_count = 0
    log_interval = max(1, nL // 5) # Log roughly every 20% progress

    if n_workers <= 0:
        if verbose >= 1:
            print(f"  * Parsing {nL} states sequentially...")
        for arg in worker_args:
            idx, G_z, E_z = _process_single_state(arg)
            G[:, :, idx] = G_z
            E[:, :, :, idx] = E_z
            completed_count += 1
            if (completed_count % log_interval == 0 or completed_count == nL) and verbose >= 2:
                print(f"    - Parsed {completed_count}/{nL} states ({(completed_count/nL)*100:.0f}%)")
    else:
        if verbose >= 1:
            print(f"  * Parsing {nL} states in parallel across {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Using as_completed instead of map to track dynamic progress and log it
            futures = {executor.submit(_process_single_state, arg): arg for arg in worker_args}
            
            for future in as_completed(futures):
                idx, G_z, E_z = future.result()
                
                G[:, :, idx] = G_z
                E[:, :, :, idx] = E_z
                
                completed_count += 1
                if (completed_count % log_interval == 0 or completed_count == nL) and verbose >= 2:
                    print(f"    - Parsed {completed_count}/{nL} states ({(completed_count/nL)*100:.0f}%)")
        
    # Calculate R and P serially (they are very fast matrix operations)
    if verbose >= 2:
        print(f"  * Computing Source (R) and Pool (P) matrices serially...")
    R_mat = np.zeros((nS, nL))
    P = np.zeros((nS, nL))
    
    # 3. Source matrix R
    for z in range(nL):
        for i_idx in range(nS):
            inflow = np.sum(G[i_idx, :, z])
            outflow = np.sum(G[:, i_idx, z])
            R_mat[i_idx, z] = inflow - outflow
            
    # 4. Pool matrix P
    for z in range(nL - 1):
        dt = t[z+1] - t[z]
        for i_idx in range(nS):
            P[i_idx, z+1] = P[i_idx, z] + dt * R_mat[i_idx, z]
            if mixing_jumps is not None and (z+1) in mixing_jumps:
                P[i_idx, z+1] += mixing_jumps[z+1][i_idx]
            
    tracked_species_names = [gas_dummy.species_names[i] for i in tracked_species_indices]
            
    if verbose >= 1:
        print(f"  [OK] Flux matrices constructed: G {G.shape}, E {E.shape}")
    logger.info(f"Parallel Cantera parsing completed successfully.")
    return G, E, R_mat, P, t, tracked_species_names, sub_mechs
