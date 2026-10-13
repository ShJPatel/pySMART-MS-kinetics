import numpy as np
import time
import logging
from numba import njit

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@njit(cache=True)
def run_smart_timestepping_jit(G, E, R, P, t, F, H, Q, U, N, tol1, tol2, max_iter, nS, nM, nL, delta_P_mix, k_mix):
    F_prev1 = np.zeros((nS, nS, nM))
    F_prev2 = np.zeros((nS, nS))
    
    for z in range(nL):
        # Physical threshold check: if total element flux is negligible, skip convergence loop
        total_flux = 0.0
        for i in range(nS):
            for j in range(nS):
                total_flux += abs(G[i, j, z])
                
        if total_flux < 1e-12:
            for i in range(nS):
                for j in range(nS):
                    for k in range(nM):
                        F[i, j, k, z] = E[i, j, k, z]
            for k in range(nM):
                for i in range(nS):
                    allocate_pool(H, N, G, F, P, Q, R, U, t, i, k, z, nS, nM, delta_P_mix, k_mix)
            continue
            
        epsilon_submechs = 1.0
        iter1 = 0
        while epsilon_submechs > tol1 and iter1 < max_iter:
            # Copy values to F_prev1 buffer
            for i in range(nS):
                for j in range(nS):
                    for k in range(nM):
                        F_prev1[i, j, k] = F[i, j, k, z]
            
            for k in range(nM):
                epsilon_submech_k = 1.0
                iter2 = 0
                while epsilon_submech_k > tol2 and iter2 < max_iter:
                    # Copy values to F_prev2 buffer
                    for i in range(nS):
                        for j in range(nS):
                            F_prev2[i, j] = F[i, j, k, z]
                    
                    for i in range(nS):
                        calculate_allocation_factor(H, N, F, G, P, Q, R, U, i, k, z, nS, nM)
                        for j in range(nS):
                            allocate_flux(H, F, G, E, i, j, k, z, nS, nM)
                            
                    num2 = 0.0
                    den2 = 0.0
                    for row_idx in range(nS):
                        for col_idx in range(nS):
                            diff_val = F[row_idx, col_idx, k, z] - F_prev2[row_idx, col_idx]
                            num2 += diff_val * diff_val
                            den2 += F[row_idx, col_idx, k, z] * F[row_idx, col_idx, k, z]
                    num2 = np.sqrt(num2)
                    den2 = np.sqrt(den2)
                    
                    epsilon_submech_k = num2 / den2 if den2 > 0 else 0.0
                    iter2 += 1
                    
            num1 = 0.0
            den1 = 0.0
            for row_idx in range(nS):
                for col_idx in range(nS):
                    for m_idx in range(nM):
                        diff_val = F[row_idx, col_idx, m_idx, z] - F_prev1[row_idx, col_idx, m_idx]
                        num1 += diff_val * diff_val
                        den1 += F[row_idx, col_idx, m_idx, z] * F[row_idx, col_idx, m_idx, z]
            num1 = np.sqrt(num1)
            den1 = np.sqrt(den1)
            
            epsilon_submechs = num1 / den1 if den1 > 0 else 0.0
            iter1 += 1
            
        for k in range(nM):
            for i in range(nS):
                allocate_pool(H, N, G, F, P, Q, R, U, t, i, k, z, nS, nM, delta_P_mix, k_mix)

def run_smart_algorithm(G, E, R, P, t, tol1=1e-5, tol2=1e-5, max_iter=5, delta_P_mix=None, k_mix=-1, verbose=1):
    """
    Executes the SMART algorithm iterative procedure to resolve
    sub-mechanism elemental allocations across species.
    
    Args:
        G (nS, nS, nL): Global element flux matrix [j->i at z]
        E (nS, nS, nM, nL): Initiation flux matrix [j->i for sub-mech k at z]
        R (nS, nL): Source matrix for species i at z
        P (nS, nL): Pool matrix for species i at z
        t (nL): Time array (or residence time derived from spatial steps)
        tol1 (float): Tolerance for epsilon1 (allocated fluxes in sub-mech)
        tol2 (float): Tolerance for epsilon2 (allocated fluxes across species)
        max_iter (int): Maximum iterations to prevent infinite loops.
        delta_P_mix (nS, nL, optional): Stage mixing jump matrix for multi-stage combustors.
        k_mix (int, optional): Sub-mechanism index for Staged Mixing.
        verbose (int, default 1): Diagnostic print verbosity (0=silent, 1=table, 2=detailed).
        
    Returns:
        F, H, Q, U, N matrices.
    """
    nS, nS_dim, nL = G.shape
    nM = E.shape[2]
    if delta_P_mix is None:
        delta_P_mix = np.zeros((nS, nL), dtype=np.float64)
    else:
        delta_P_mix = np.ascontiguousarray(delta_P_mix, dtype=np.float64)

    if verbose >= 1:
        print(f"[pySMART:Solver] Initializing SMART Serial Solver:")
        print(f"  * Grid points: {nL:,} | Species: {nS} | Sub-mechanisms: {nM}")
        print(f"  * Iterative convergence (max {max_iter} iters, tol={tol1:.1e}):")
        print(f"    {'Iter':>4} | {'Max Residual':>12} | {'eps_F':>10} | {'eps_H':>10} | {'eps_Q':>10} | {'Time':>7}")
        print(f"    {'-'*4}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

    logger.info(f"Starting SMART Algorithm with {nS} species, {nM} sub-mechanisms, and {nL} spatial/temporal steps.")
    start_time_total = time.time()

    # Initialize F, H, Q, U outside the loop so they exist for the residual calculation
    # but the pseudocode explicitly resets them inside the loop to E and zeros.
    F = E.copy()
    H = np.zeros((nS, nM, nL))
    Q = np.zeros((nS, nM, nL))
    U = np.zeros((nS, nM, nL))

    # Initialize Flux Correction Factor N to 1/nM for all i, k
    # (Uniform distribution)
    N = np.ones((nS, nM)) / nM

    epsilon = 1.0
    tau = tol1

    iter_count = 0
    # Executes nested iterative convergence of flux allocations
    while epsilon > tau and iter_count < max_iter:
        loop_start_time = time.time()

        # Save previous state
        # The residual epsilon is calculated using the updated matrices H, F, and Q 
        # along with results from the previous iteration (H~, F~, Q~).
        F_prev = F.copy()
        H_prev = H.copy()
        Q_prev = Q.copy()

        # Pseudocode Main Loop Initialization
        F = E.copy()
        H = np.zeros((nS, nM, nL))
        Q = np.zeros((nS, nM, nL))
        U = np.zeros((nS, nM, nL))

        logger.info(f"--- Main Iteration {iter_count + 1} ---")

        # Run JIT compiled time stepping loop
        run_smart_timestepping_jit(G, E, R, P, t, F, H, Q, U, N, tol1, tol2, max_iter, nS, nM, nL, delta_P_mix, k_mix)

        # Update Flux Correction Factor (nu_{i,k})
        logger.debug("  Updating Flux Correction Factors (N)...")
        for k in range(nM):
            for i in range(nS):
                # integral sum_j f_{i,j,k} dt
                num_integrand = np.sum(F[i, :, k, :], axis=0)
                num_integral = np.trapz(num_integrand, t)

                # integral sum_j g_{i,j} dt
                den_integrand = np.sum(G[i, :, :], axis=0)
                den_integral = np.trapz(den_integrand, t)

                N[i, k] = float(num_integral) / float(den_integral) if den_integral > 0 and not np.isnan(den_integral) and not np.isinf(den_integral) else N[i,k]

        # Overall residual
        logger.debug("  Calculating overall residuals...")
        norm_F = np.linalg.norm(F)
        norm_H = np.linalg.norm(H)
        norm_Q = np.linalg.norm(Q)

        num_F = np.linalg.norm(F - F_prev)
        num_H = np.linalg.norm(H - H_prev)
        num_Q = np.linalg.norm(Q - Q_prev)

        # Add guard if norm is inf or nan
        eps_F = float(num_F) / float(norm_F) if norm_F > 0  and not np.isnan(norm_F) and not np.isinf(norm_F) else 0.0
        eps_H = float(num_H) / float(norm_H) if norm_H > 0 and not np.isnan(norm_H) and not np.isinf(norm_H)  else 0.0
        eps_Q = float(num_Q) / float(norm_Q) if norm_Q > 0 and not np.isnan(norm_Q) and not np.isinf(norm_Q) else 0.0

        epsilon = max(eps_F, eps_H, eps_Q)

        loop_time = time.time() - loop_start_time
        iter_count += 1

        if verbose >= 1:
            print(f"    {iter_count:>4} | {epsilon:>12.3e} | {eps_F:>10.2e} | {eps_H:>10.2e} | {eps_Q:>10.2e} | {loop_time:>6.2f}s")

        logger.info(f"Main loop {iter_count} complete in {loop_time:.2f}s | "
                    f"Overall Epsilon: {epsilon:.6e} "
                    f"(eps_F={eps_F:.2e}, eps_H={eps_H:.2e}, eps_Q={eps_Q:.2e})")

    total_time = time.time() - start_time_total
    status = "converged" if epsilon <= tau else f"stopped (max_iter={max_iter})"
    if verbose >= 1:
        print(f"  [OK] Serial SMART solve {status} | Final residual: {epsilon:.2e} | Total solve time: {total_time:.2f}s")
    if epsilon <= tau:
        logger.info(f"SMART Algorithm converged in {iter_count} iterations and {total_time:.2f}s.")
    else:
        logger.warning(
            f"SMART Algorithm reached max_iter ({max_iter}) without full convergence. Final epsilon: {epsilon:.6e}")

    return F, H, Q, U, N


@njit(cache=True)
def calculate_allocation_factor(H, N, F, G, P, Q, R, U, i, k, z, nS, nM):
    """
    Computes eta_{i,k,z} for species Si, sub-mechanism Mk, at flame position Lz.
    """
    g_in = 0.0
    for j in range(nS):
        g_in += G[i, j, z]
        
    g_out = 0.0
    for j in range(nS):
        g_out += G[j, i, z]

    if g_in >= g_out:
        alpha = 0
    else:
        alpha = 1
        if P[i, z] > 0:
            sum_q = 0.0
            for k_idx in range(nM):
                val = Q[i, k_idx, z]
                if val > 0:
                    sum_q += val
                    
            if sum_q > 0:
                q_val = Q[i, k, z]
                if q_val < 0:
                    q_val = 0.0
                U[i, k, z] = (q_val / sum_q) * R[i, z]
            else:
                sum_n = 0.0
                for k_idx in range(nM):
                    sum_n += N[i, k_idx]
                if sum_n > 0:
                    U[i, k, z] = (N[i, k] / sum_n) * R[i, z]
                else:
                    U[i, k, z] = (1.0 / nM) * R[i, z]
        else:
            sum_n = 0.0
            for k_idx in range(nM):
                sum_n += N[i, k_idx]
            if sum_n > 0:
                U[i, k, z] = (N[i, k] / sum_n) * R[i, z]
            else:
                U[i, k, z] = (1.0 / nM) * R[i, z]

    f_sum = 0.0
    for j in range(nS):
        f_sum += F[i, j, k, z]
        
    # The corrected formula that aligns with the physical definition of "total inflow"
    # and prevents mathematically incorrect expansion/contraction of terms
    # when r_i is defined as (g_in - g_out), which is negative for a source.
    num = f_sum - alpha * U[i, k, z]
    den = g_in - alpha * R[i, z]

    if abs(den) > 0:
        val = num / den
        if val > 1.0:
            H[i, k, z] = 1.0
        elif val < 0.0:
            H[i, k, z] = 0.0
        else:
            H[i, k, z] = val
    else:
        H[i, k, z] = 0.0

@njit(cache=True)
def allocate_flux(H, F, G, E, i, j, k, z, nS, nM):
    """
    Allocates global flux between Si and Sj for sub-mechanism Mk at flame position Lz.
    We only process each undirected pair once (i < j) to compute the net flux and avoid clobbering.
    """
    if i >= j:
        return

    # Net global flux from i to j
    net_g = G[j, i, z] - G[i, j, z]
    
    # Net initiation flux from i to j for sub-mech k
    net_e_k = E[j, i, k, z] - E[i, j, k, z]
    
    # Total net initiation flux from i to j across all sub-mechs
    sum_net_e = 0.0
    for k_idx in range(nM):
        sum_net_e += (E[j, i, k_idx, z] - E[i, j, k_idx, z])

    if net_g == 0 and sum_net_e == 0:
        return

    # Allocatable flow from i to j
    diff = net_g - sum_net_e

    # Determine which allocation factor to use
    if diff > 0:
        # Allocatable flow is i -> j
        eta = H[i, k, z]
    elif diff < 0:
        # Allocatable flow is j -> i
        eta = H[j, k, z]
    else:
        eta = 0.0

    # Ultimate net flux from i to j for sub-mech k
    net_f_k = eta * diff + net_e_k

    if net_f_k > 0:
        # Net flux is i -> j
        F[j, i, k, z] = net_f_k
        F[i, j, k, z] = 0.0
    elif net_f_k < 0:
        # Net flux is j -> i
        F[i, j, k, z] = -net_f_k
        F[j, i, k, z] = 0.0
    else:
        F[j, i, k, z] = 0.0
        F[i, j, k, z] = 0.0

@njit(cache=True)
def allocate_pool(H, N, G, F, P, Q, R, U, t, i, k, z, nS, nM, delta_P_mix, k_mix):
    """
    Computes contribution to pool of Si from sub-mechanism Mk at flame position Lz.
    Includes exact stage mixing jump tracking for multi-stage combustors.
    """
    # Calculate net source flux for this sub-mechanism by comparing
    # its total allocated inflow and outflow for the species.
    in_k = 0.0
    for j in range(nS):
        in_k += F[i, j, k, z]
    out_k = 0.0
    for j in range(nS):
        out_k += F[j, i, k, z]
        
    U[i, k, z] = in_k - out_k

    nL = len(t)
    if z + 1 < nL:
        dt = t[z + 1] - t[z]
        q_next = Q[i, k, z] + dt * U[i, k, z]
        
        # Check for stage mixing jump at z + 1
        dp = delta_P_mix[i, z + 1]
        if dp != 0.0:
            if dp > 0.0:
                # Injection: attribute positive addition to designated mixing sub-mechanism
                if k == k_mix or (k_mix < 0 and k == 0):
                    q_next += dp
            else:
                # Dilution: scale existing sub-mechanism pool proportionally
                p_curr = P[i, z]
                if abs(p_curr) > 1e-30:
                    q_next += (Q[i, k, z] / p_curr) * dp
                else:
                    q_next += (1.0 / nM) * dp
                    
        Q[i, k, z + 1] = q_next
