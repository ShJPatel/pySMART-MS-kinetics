"""
smart_core_multigrid.py

Hierarchical Domain Decomposition SMART Algorithm.

Architecture:
    Level 0 (Coarse Warmup, serial):
        1 full forward SMART pass over all z=0..N_L.
        Initialises ν_ik (N matrix) much better than uniform 1/n_M.

    Level 1 (Parallel Sub-Domain Passes):
        z split into n_chunks.  Each chunk runs the SAME inner JIT as the
        original smart_core, but only over its slice G_c, E_c, R_c, P_c.
        Starting pool P[:,z_start] is already correct because cantera_parser
        pre-computes the cumulative pool over the entire trajectory before
        SMART runs — no boundary communication for the pool state.
        Only the ν_ik matrix (n_S × n_M floats) is shared globally.

    Level 2 (Global ν_ik Synchronisation):
        ν_ik = Σ_chunks ∫ Σ_j F_{i,j,k} dt  /  Σ_chunks ∫ Σ_j G_{i,j} dt
        Both integrals are additive across disjoint z-intervals.
        Compute global residual ε, repeat Level 1 until converged.

Key insight:
    The P matrix from cantera_parser.py is P[i,z] = ∫₀^{t_z} R_i dt′ (Euler
    forward integral).  So P[:,z_start] is already the exact cumulative pool
    at the chunk boundary — no explicit state communication is required.
    Only ν_ik (a tiny n_S×n_M matrix) needs global synchronisation per
    Level-2 iteration.  Communication overhead is completely negligible.

Speedup analysis:
    Serial SMART:   O(N_L · n_S² · n_M · n_iter)
    Multigrid L1:   O(N_L/P · n_S² · n_M · n_iter_inner)   × P parallel
                  + O(N_L · n_S² · n_M · 1)                (Level-0 warmup)
    Level-0 costs ~10% overhead; Level-1 scales nearly linearly with P
    workers up to the point where process-spawn overhead dominates
    (expected to occur around P ≈ 8-16 for chunks of size ~30-50 points).
"""

import time
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in chunk JIT — identical inner loop from smart_core but on a z-slice
# ─────────────────────────────────────────────────────────────────────────────

def _run_smart_slice_serial(G_c, E_c, R_c, P_c, t_c, N,
                             tol1, tol2, max_iter_inner,
                             delta_P_mix_c, k_mix, Q_init_c=None):
    """
    Runs the SMART inner convergence loop over a z-slice [z_start, z_end).
    Calls the ORIGINAL JIT functions from smart_core directly so that
    allocate_flux / calculate_allocation_factor / allocate_pool are identical
    to the validated serial implementation.
    """
    from .solver_serial import run_smart_timestepping_jit

    nS, _, nL_c = G_c.shape
    nM = E_c.shape[2]

    F_c = E_c.copy()
    H_c = np.zeros((nS, nM, nL_c))
    Q_c = np.zeros((nS, nM, nL_c))
    if Q_init_c is not None:
        Q_c[:, :, 0] = Q_init_c
    U_c = np.zeros((nS, nM, nL_c))

    run_smart_timestepping_jit(G_c, E_c, R_c, P_c, t_c,
                               F_c, H_c, Q_c, U_c,
                               N, tol1, tol2, max_iter_inner,
                               nS, nM, nL_c, delta_P_mix_c, k_mix)
    return F_c, H_c, Q_c, U_c


def _chunk_worker(args):
    """
    Process-pool worker.  Returns chunk index + all local matrices + ν_ik integrals.
    Must be top-level (not a closure) for pickling.
    """
    (c_idx, G_c, E_c, R_c, P_c, t_c, N, tol1, tol2, max_iter_inner, delta_P_mix_c, k_mix, Q_init_c) = args
    nS, _, nL_c = G_c.shape
    nM = E_c.shape[2]

    F_c, H_c, Q_c, U_c = _run_smart_slice_serial(
        G_c, E_c, R_c, P_c, t_c, N, tol1, tol2, max_iter_inner, delta_P_mix_c, k_mix, Q_init_c
    )

    # Local integrals for ν_ik update — additive across chunks
    # local_F_num[i,k] = ∫_chunk Σ_j F[i,j,k](t) dt
    # local_G_den[i]   = ∫_chunk Σ_j G[i,j](t)   dt
    local_F_num = np.zeros((nS, nM))
    local_G_den = np.zeros(nS)
    for i in range(nS):
        for k in range(nM):
            integrand = np.sum(F_c[i, :, k, :], axis=0)
            local_F_num[i, k] = np.trapz(integrand, t_c)
        local_G_den[i] = np.trapz(np.sum(G_c[i, :, :], axis=0), t_c)

    return c_idx, F_c, H_c, Q_c, U_c, local_F_num, local_G_den


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _update_nu(F_num_total, G_den_total, N_prev):
    """Update ν_ik from summed global integrals; preserves previous value on zero denominator."""
    N_new = N_prev.copy()
    nS, nM = N_prev.shape
    for i in range(nS):
        den = float(G_den_total[i])
        if den > 0 and not (np.isnan(den) or np.isinf(den)):
            for k in range(nM):
                N_new[i, k] = float(F_num_total[i, k]) / den
    return N_new


def _safe_rel_norm(A, B):
    num = np.linalg.norm(A - B)
    den = np.linalg.norm(A)
    if den > 0 and not np.isnan(den) and not np.isinf(den):
        return float(num / den)
    return 0.0


def _make_chunks(nL, n_chunks, min_chunk=30, stage_slices=None):
    """
    Returns list of (z_start, z_end) pairs covering [0, nL).
    If stage_slices is provided (dict or list of (start, end)),
    chunks each stage independently to guarantee no chunk crosses a stage boundary.
    """
    if stage_slices is not None:
        slices_list = list(stage_slices.values()) if isinstance(stage_slices, dict) else stage_slices
        slices_list = sorted(slices_list, key=lambda s: s[0])
        bounds = []
        for (s_start, s_end) in slices_list:
            s_len = s_end - s_start
            if s_len <= 0:
                continue
            stage_chunks = max(1, round(n_chunks * (s_len / nL)))
            stage_bounds = _make_chunks(s_len, stage_chunks, min_chunk=min_chunk)
            for (bs, be) in stage_bounds:
                bounds.append((bs + s_start, be + s_start))
        return bounds

    actual_chunks = max(1, min(n_chunks, nL // max(1, min_chunk // 2)))
    splits = np.array_split(np.arange(nL), actual_chunks)
    bounds = [(int(s[0]), int(s[-1]) + 1) for s in splits if len(s) > 0]
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def determine_optimal_partition(nL: int, max_workers: int = None, target_chunk_size: int = 50, min_chunk_size: int = 30):
    """
    Dynamically determines the optimal number of chunks and workers for trajectory length nL.
    
    Logic:
        1. Calculates ideal number of chunks to keep chunks around ~50 points (preventing flame-front
           over-partitioning while minimizing parallel compute time).
        2. Sets worker count to match chunk count up to the hardware CPU limit (n_workers = min(cpus, n_chunks)).
           - For compact grids (nL ~ 300): uses ~6-8 workers with zero starvation.
           - For large grids (nL ~ 2,500+): automatically scales up to 20-28 workers.

    Parameters:
        nL: int
            Total number of grid points in the chemical trajectory.
        max_workers: int, optional
            Explicit worker count override. If None, uses max(1, multiprocessing.cpu_count() - 2).
        target_chunk_size: int, default=50
            Target points per chunk for optimal compute-to-IPC efficiency.
        min_chunk_size: int, default=30
            Absolute lower bound on chunk size.

    Returns:
        n_chunks: int
            Number of spatial/temporal chunks to construct.
        n_workers: int
            Optimal worker pool concurrency matching chunk count.
    """
    import multiprocessing
    avail_cpus = max(1, multiprocessing.cpu_count() - 2)
    limit_workers = avail_cpus if max_workers is None else max_workers

    # 1. Ideal chunks based on target chunk size (~50 points)
    ideal_chunks = max(1, nL // target_chunk_size)
    
    # 2. Match chunks to available hardware limit
    n_chunks = min(ideal_chunks, limit_workers)
    
    # 3. Ensure chunk sizes don't drop below the safety floor (min_chunk_size)
    while n_chunks > 1 and (nL // n_chunks) < min_chunk_size:
        n_chunks -= 1
        
    # 4. Set worker pool size exactly equal to chunk count so no worker runs sequentially
    n_workers = min(n_chunks, limit_workers)
    return n_chunks, n_workers


def run_smart_algorithm_multigrid(G, E, R, P, t,
                                   n_chunks: int = None,
                                   tol1: float = 1e-5,
                                   tol2: float = 1e-5,
                                   max_iter_inner: int = 5,
                                   max_iter_level2: int = 5,
                                   min_chunk_size: int = 30,
                                   target_chunk_size: int = 50,
                                   n_workers: int = None,
                                   delta_P_mix: np.ndarray = None,
                                   k_mix: int = -1,
                                   stage_slices: dict = None,
                                   return_iteration_history: bool = False,
                                   tracked_species_idx: int = None):
    """
    Hierarchical multi-grid SMART algorithm with automatic dynamic chunking and multi-stage mixing support.

    Args:
        G:  (nS, nS, nL)       Global element flux matrix.
        E:  (nS, nS, nM, nL)   Initiation flux matrix.
        R:  (nS, nL)            Net source matrix.
        P:  (nS, nL)            Pre-integrated cumulative pool matrix.
        t:  (nL,)               Time array.
        n_chunks:           Number of parallel chunks (None for automatic dynamic allocation).
        tol1, tol2:         Inner convergence tolerances.
        max_iter_inner:     Max inner JIT iterations per chunk per L1 call.
        max_iter_level2:    Max global Level-2 iterations.
        min_chunk_size:     Minimum points per chunk (default 30).
        target_chunk_size:  Ideal target points per chunk for dynamic sizing (default 50).
        n_workers:          Process pool size (None for automatic dynamic allocation).
        delta_P_mix:        (nS, nL) Stage mixing jump matrix for multi-stage combustors.
        k_mix:              Sub-mechanism index for Staged Mixing.
        stage_slices:       Dict or list of stage slices to ensure clean chunk boundary alignment.
        return_iteration_history: If True, returns (F, H, Q, U, N, history_dict).
        tracked_species_idx: Specific species index to snapshot integrated yields in history (e.g. i_NO).

    Returns:
        F, H, Q, U, N (and history_dict if return_iteration_history=True)
    """
    import multiprocessing
    from .solver_serial import run_smart_timestepping_jit

    nS, _, nL = G.shape
    nM = E.shape[2]
    if delta_P_mix is None:
        delta_P_mix = np.zeros((nS, nL), dtype=np.float64)
    else:
        delta_P_mix = np.ascontiguousarray(delta_P_mix, dtype=np.float64)

    # Dynamic partition sizing if not explicitly supplied
    if n_chunks is None or n_workers is None:
        auto_chunks, auto_workers = determine_optimal_partition(
            nL, max_workers=n_workers, target_chunk_size=target_chunk_size, min_chunk_size=min_chunk_size
        )
        if n_chunks is None:
            n_chunks = auto_chunks
        if n_workers is None:
            n_workers = auto_workers

    logger.info(f"Multigrid SMART (Dynamic Sizing) | nS={nS} nM={nM} nL={nL} | "
                f"n_chunks={n_chunks} (avg {nL//n_chunks} pts/chk) | n_workers={n_workers}")
    t_total = time.time()

    # ── Level 0: 1-pass serial warmup ────────────────────────────────────────
    logger.info("Level-0 warmup: 1 serial pass to initialise ν_ik…")
    t0 = time.time()

    N = np.ones((nS, nM)) / nM          # uniform start
    F = E.copy()
    H = np.zeros((nS, nM, nL))
    Q = np.zeros((nS, nM, nL))
    U = np.zeros((nS, nM, nL))

    run_smart_timestepping_jit(G, E, R, P, t, F, H, Q, U,
                               N, tol1, tol2, 3, nS, nM, nL, delta_P_mix, k_mix)

    # Compute ν_ik from warmup integrals
    F_num_total = np.zeros((nS, nM))
    G_den_total = np.zeros(nS)
    for i in range(nS):
        for k in range(nM):
            F_num_total[i, k] = np.trapz(np.sum(F[i, :, k, :], axis=0), t)
        G_den_total[i] = np.trapz(np.sum(G[i, :, :], axis=0), t)
    N = _update_nu(F_num_total, G_den_total, N)
    t_warmup = time.time() - t0
    logger.info(f"Level-0 complete in {t_warmup:.2f}s")

    # Compute chunk boundaries (respecting stage boundaries if provided)
    bounds = _make_chunks(nL, n_chunks, min_chunk=min_chunk_size, stage_slices=stage_slices)
    n_actual = len(bounds)
    logger.info(f"Chunks: {[(s, e, e-s) for s, e in bounds]}")

    # Global output arrays (assembled from chunks)
    F_global = np.zeros_like(F)
    H_global = np.zeros_like(H)
    Q_global = np.zeros_like(Q)
    U_global = np.zeros_like(U)
    Q_current = Q.copy() # Continuous pool initialized from Level-0 warmup

    epsilon = 1.0
    level2_iter = 0
    history = {}

    if return_iteration_history:
        # Snapshot Level-0 state
        sp_idx = tracked_species_idx if tracked_species_idx is not None else 0
        l0_yields = [float(np.trapz(U[sp_idx, k, :], t)) for k in range(nM)]
        history[0] = {
            'level': 0,
            'epsilon': 1.0,
            'elapsed_s': t_warmup,
            'integrated_yields': l0_yields,
            'nu_ik': N.copy()
        }

    # ── Level-2 loop: synchronise ν_ik across chunks ─────────────────────────
    # Keep pool alive across all Level-2 iterations to avoid repetitive spawning overhead
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    try:
        while epsilon > tol1 and level2_iter < max_iter_level2:
            t_l2 = time.time()
            F_prev = F_global.copy()
            H_prev = H_global.copy()
            Q_prev = Q_global.copy()

            # ── Level-1: parallel chunk solves ────────────────────────────────────
            chunk_args = []
            for c_idx, (z_s, z_e) in enumerate(bounds):
                chunk_args.append((
                    c_idx,
                    np.ascontiguousarray(G[:, :, z_s:z_e]),
                    np.ascontiguousarray(E[:, :, :, z_s:z_e]),
                    np.ascontiguousarray(R[:, z_s:z_e]),
                    np.ascontiguousarray(P[:, z_s:z_e]),
                    np.ascontiguousarray(t[z_s:z_e]),
                    N.copy(),       # read-only inside worker
                    tol1, tol2, max_iter_inner,
                    np.ascontiguousarray(delta_P_mix[:, z_s:z_e]),
                    k_mix,
                    np.ascontiguousarray(Q_current[:, :, z_s])
                ))

            results = [None] * n_actual
            if pool is None:
                for arg in chunk_args:
                    res = _chunk_worker(arg)
                    results[res[0]] = res
            else:
                futs = {pool.submit(_chunk_worker, a): a for a in chunk_args}
                for fut in as_completed(futs):
                    res = fut.result()
                    results[res[0]] = res

            # Assemble global arrays & accumulate nu_ik integrals
            F_num_total = np.zeros((nS, nM))
            G_den_total = np.zeros(nS)
            for c_idx, (z_s, z_e) in enumerate(bounds):
                _, F_c, H_c, Q_c, U_c, lF, lG = results[c_idx]
                F_global[:, :, :, z_s:z_e] = F_c
                H_global[:, :, z_s:z_e]    = H_c
                Q_global[:, :, z_s:z_e]    = Q_c
                U_global[:, :, z_s:z_e]    = U_c
                F_num_total += lF
                G_den_total += lG

            Q_current = Q_global.copy()
            N = _update_nu(F_num_total, G_den_total, N)

            # Level-2 residual
            eps_F = _safe_rel_norm(F_global, F_prev)
            eps_H = _safe_rel_norm(H_global, H_prev)
            eps_Q = _safe_rel_norm(Q_global, Q_prev)
            epsilon = max(eps_F, eps_H, eps_Q)
            level2_iter += 1
            iter_duration = time.time() - t_l2

            if return_iteration_history:
                sp_idx = tracked_species_idx if tracked_species_idx is not None else 0
                iter_yields = [float(np.trapz(U_global[sp_idx, k, :], t)) for k in range(nM)]
                history[level2_iter] = {
                    'level': level2_iter,
                    'epsilon': float(epsilon),
                    'eps_F': float(eps_F),
                    'eps_H': float(eps_H),
                    'eps_Q': float(eps_Q),
                    'elapsed_s': time.time() - t_total,
                    'iter_duration_s': iter_duration,
                    'integrated_yields': iter_yields,
                    'nu_ik': N.copy()
                }

            logger.info(f"Level-2 iter {level2_iter}: eps={epsilon:.3e} "
                        f"(F={eps_F:.2e} H={eps_H:.2e} Q={eps_Q:.2e}) "
                        f"in {iter_duration:.2f}s")
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    elapsed = time.time() - t_total
    status = "converged" if epsilon <= tol1 else f"stopped at max_iter_level2={max_iter_level2}"
    logger.info(f"Multigrid SMART {status} | eps={epsilon:.3e} | "
                f"{level2_iter} L2 iters | {elapsed:.2f}s total")

    if return_iteration_history:
        return F_global, H_global, Q_global, U_global, N, history
    return F_global, H_global, Q_global, U_global, N


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in wrapper for smart_core.run_smart_algorithm
# ─────────────────────────────────────────────────────────────────────────────

def run_smart_algorithm(G, E, R, P, t,
                        tol1=1e-5, tol2=1e-5, max_iter=5,
                        n_chunks: int = None, n_workers: int = None,
                        target_chunk_size: int = 45):
    """
    Drop-in replacement for smart_core.run_smart_algorithm using multigrid
    with automatic dynamic chunking and worker sizing.
    """
    return run_smart_algorithm_multigrid(
        G, E, R, P, t,
        n_chunks=n_chunks, tol1=tol1, tol2=tol2,
        max_iter_inner=max_iter, max_iter_level2=max_iter,
        n_workers=n_workers, target_chunk_size=target_chunk_size
    )
