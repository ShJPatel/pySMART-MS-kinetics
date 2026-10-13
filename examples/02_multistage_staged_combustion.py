"""
Example 2: Multi-Stage Axially Staged Combustor Analysis (SMART-MS)
Mirrors the canonical NRRL axially staged combustor demonstration:
- Stage 1: Stoichiometric primary flame products (CH4/air, Tin=600 K, P=10 atm)
- Stage Boundary: Discrete mixing jump tracking (50% staged fuel injection)
- Stage 2: Rich relaxation stage (Infinite-rate reactor, target_time=5 ms)
- Multi-target tracking: Resolves both NO destruction and intermediate cyanide (HCN) formation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def simulate_nrrl_staged_combustor():
    """
    Simulates a 2-stage staged combustor trajectory matching NRRL.Combustor conditions:
    P = 10 atm, T_inlet = 600 K, frac_m_fuel_staged = 0.5 (50% fuel staged).
    """
    ct.suppress_thermo_warnings()
    mech = "gri30.yaml"
    gas = ct.Solution(mech)
    P_val = 10.0 * ct.one_atm
    T_in = 600.0

    print("   Setting up Stage 1: Stoichiometric Core Combustion (Tin=600K, P=10atm)...")
    gas.TP = 1100.0, P_val  # Ignition start temperature
    gas.set_equivalence_ratio(1.0, "CH4:1", "O2:1, N2:3.76")
    
    # Equilibrium stopping state
    gas_eq = ct.Solution(mech)
    gas_eq.TP = T_in, P_val
    gas_eq.set_equivalence_ratio(1.0, "CH4:1", "O2:1, N2:3.76")
    gas_eq.equilibrate("HP")

    r1 = ct.IdealGasConstPressureReactor(gas)
    sim1 = ct.ReactorNet([r1])

    times, temps, pressures, stages = [], [], [], []
    species_X = {sp: [] for sp in gas.species_names}

    t_prev = 0.0
    step_cnt = 0
    while sim1.time < 0.05:
        sim1.step()
        dt = sim1.time - t_prev
        t_prev = sim1.time
        step_cnt += 1
        if step_cnt % 5 == 0 or (gas.T >= (gas_eq.T - 20.0) and dt >= 0.002):
            times.append(sim1.time)
            temps.append(gas.T)
            pressures.append(gas.P)
            stages.append(1)
            for i, sp in enumerate(gas.species_names):
                species_X[sp].append(gas.X[i])
        if gas.T >= (gas_eq.T - 20.0) and dt >= 0.002:
            break

    t1_end = times[-1]
    print(f"   Stage 1 Complete: {len(times)} states, exit T={temps[-1]:.1f} K, exit NO={species_X['NO'][-1]*1e6:.1f} ppm")

    # Stage 2: Secondary Staged Fuel Injection (50% staged fuel diverted to rich zone)
    print("   Simulating Stage Boundary Injection: Adding 50% staged CH4...")
    fuel = ct.Solution(mech)
    fuel.TPX = T_in, P_val, "CH4:1"
    
    far = 1.0 / gas.stoich_air_fuel_ratio("CH4:1", "O2:1, N2:3.76")
    m_fuel_stoich = 1.0 * far / (1.0 + far)
    m_fuel_rr = m_fuel_stoich  # 50% staged fuel fraction

    mixed = ct.Quantity(gas, mass=1.0, constant="HP") + ct.Quantity(fuel, mass=m_fuel_rr, constant="HP")
    gas_rich = mixed.phase

    r2 = ct.IdealGasConstPressureReactor(gas_rich)
    sim2 = ct.ReactorNet([r2])

    print("   Setting up Stage 2: Rich Relaxation (target_time = 5 ms)...")
    step_cnt = 0
    while sim2.time < 0.005:
        sim2.step()
        step_cnt += 1
        if step_cnt % 3 == 0 or sim2.time >= 0.005:
            times.append(t1_end + sim2.time)
            temps.append(gas_rich.T)
            pressures.append(gas_rich.P)
            stages.append(2)
            for i, sp in enumerate(gas_rich.species_names):
                species_X[sp].append(gas_rich.X[i])

    print(f"   Stage 2 Complete: exit T={temps[-1]:.1f} K, exit NO={species_X['NO'][-1]*1e6:.1f} ppm")

    df = pd.DataFrame({"time": times, "T": temps, "P": pressures, "stage": stages})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals

    return df, t1_end

def main():
    print("1. Generating NRRL Axially Staged Combustor Trajectory...")
    df, t_stage1_end = simulate_nrrl_staged_combustor()
    print(f"   Total composite trajectory: {len(df)} points across 2 stages.")

    # Zero-shift time relative to the stage interface (t=0 at secondary fuel injection)
    df["time_rel_boundary"] = df["time"] - t_stage1_end

    print("\n2. Executing pySMART Multi-Stage Jump Tracking (SMART-MS)...")
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time_rel_boundary",
        tracked_element="N",
        target_species="NO",
        stage_col="stage",    # Enables discrete stage mass-jump tracking
        multigrid=True,
        n_workers=4
    )

    print("\n3. Identified Stage Slices:")
    for st_name, s_slice in result.stage_slices.items():
        print(f"   Stage {st_name}: rows {s_slice.start} to {s_slice.stop}")

    print("\n4. Integrated Cumulative Yields for NO:")
    df_no = result.get_cumulative_yields(target="NO")
    print(df_no.to_string(index=False))

    print("\n5. Integrated Cumulative Yields for Intermediate HCN:")
    df_hcn = result.get_cumulative_yields(target="HCN")
    print(df_hcn.to_string(index=False))

    print("\n6. Generating diagnostic plots zoomed in on stage interface & rich reaction zone...")
    # Zoom limits: [-0.3 ms, 1.2 ms] around stage boundary
    # Shows late Stage 1 flame exit followed by instantaneous mixing jump and rich reburning!
    zoom_xlim = (-0.3, 1.2)

    # Plot NO rates
    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=True)
    ax_rates.set_xlim(zoom_xlim)
    ax_rates.axvline(0.0, color="gray", linestyle=":", label="Stage Boundary (t=0)")
    ax_rates.set_title("Allocated NO Rates Across Stage Interface (SMART-MS)")
    ax_rates.legend(frameon=True)
    fig_rates.savefig("examples/02_multistage_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_rates_no.png")

    # Plot NO pools
    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=True)
    ax_pools.set_xlim(zoom_xlim)
    ax_pools.axvline(0.0, color="gray", linestyle=":", label="Stage Boundary (t=0)")
    ax_pools.set_title("Cumulative NO Pools Across Stage Interface (SMART-MS)")
    ax_pools.legend(frameon=True)
    fig_pools.savefig("examples/02_multistage_pools_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_pools_no.png")

    # Plot HCN rates (mirroring NRRL splt.plot_results_f target_sp='HCN')
    fig_hcn, ax_hcn = result.plot_rates(target="HCN", t_ms=True)
    ax_hcn.set_xlim(zoom_xlim)
    ax_hcn.axvline(0.0, color="gray", linestyle=":", label="Stage Boundary (t=0)")
    ax_hcn.set_title("Allocated HCN Rates Across Stage Interface (SMART-MS)")
    ax_hcn.legend(frameon=True)
    fig_hcn.savefig("examples/02_multistage_rates_hcn.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_rates_hcn.png")

    print("\nExample 2 finished successfully!")

if __name__ == "__main__":
    main()
