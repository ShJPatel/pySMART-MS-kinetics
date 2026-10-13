"""
Example 2: Multi-Stage Staged Combustor Analysis (SMART-MS)
Demonstrates pySMART on a classic Rich-Quench-Lean (RQL) staged combustion system:
- Stage 1 (Rich Zone): Primary fuel-rich combustion (phi = 1.35) forming rich NO intermediates.
- Interface (Quench / Secondary Air Mixing): Discrete air injection dilutes phi to 0.70.
  SMART-MS automatically attributes incoming oxidizer to 'Staged Mixing' and tracks exact mass jumps.
- Stage 2 (Lean Burnout): Secondary lean combustion (phi = 0.70) burning unburnt CO and forming lean NO.

Completely decoupled from any external combustor frameworks: built purely with standard Cantera.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def simulate_rql_combustor():
    """
    Simulates a 2-stage Rich-Quench-Lean (RQL) combustor trajectory using pure Cantera:
    - Stage 1: Rich combustion (phi = 1.35, P = 5 atm, residence time = 2.0 ms)
    - Mixing Interface: Rapid secondary air injection diluting phi to 0.70
    - Stage 2: Lean burnout (phi = 0.70, P = 5 atm, residence time = 2.0 ms)
    """
    ct.suppress_thermo_warnings()
    mech = "gri30.yaml"
    P = 5.0 * ct.one_atm

    print("1. Simulating Stage 1: Rich Combustion Zone (phi = 1.35, P = 5 atm)...")
    gas = ct.Solution(mech)
    gas.TP = 1400.0, P
    gas.set_equivalence_ratio(1.35, "CH4:1", "O2:1, N2:3.76")

    r1 = ct.IdealGasConstPressureReactor(gas)
    sim1 = ct.ReactorNet([r1])

    times, temps, pressures, stages = [], [], [], []
    species_X = {sp: [] for sp in gas.species_names}

    step_cnt = 0
    while sim1.time < 0.002:  # 2.0 ms rich stage
        sim1.step()
        step_cnt += 1
        if step_cnt % 3 == 0 or sim1.time >= 0.002:
            times.append(sim1.time)
            temps.append(gas.T)
            pressures.append(gas.P)
            stages.append(1)
            for i, sp in enumerate(gas.species_names):
                species_X[sp].append(gas.X[i])

    t1_end = times[-1]
    print(f"   Stage 1 Complete: exit T = {temps[-1]:.1f} K, exit NO = {species_X['NO'][-1]*1e6:.1f} ppm, CO = {species_X['CO'][-1]*100:.2f}%")

    print("\n2. Simulating Quench Interface: Secondary Air Injection (phi = 1.35 -> 0.70)...")
    air = ct.Solution(mech)
    air.TPX = 650.0, P, "O2:0.21, N2:0.79"

    # Compute stoichiometric air/fuel ratio and required secondary air mass
    AF_stoich = gas.stoich_air_fuel_ratio("CH4:1", "O2:0.21, N2:0.79")
    m_fuel = 1.0 / (1.0 + AF_stoich / 1.35)
    m_air_needed = m_fuel * (AF_stoich / 0.70)
    m_air_add = m_air_needed - (1.0 - m_fuel)

    # Constant enthalpy & pressure mixing
    mixed = ct.Quantity(gas, mass=1.0, constant="HP") + ct.Quantity(air, mass=m_air_add, constant="HP")
    gas_lean = mixed.phase

    print(f"   Mixing Interface: exit T = {gas_lean.T:.1f} K, NO after air dilution = {gas_lean['NO'].X[0]*1e6:.1f} ppm")

    print("\n3. Simulating Stage 2: Lean Burnout Zone (phi = 0.70, P = 5 atm)...")
    r2 = ct.IdealGasConstPressureReactor(gas_lean)
    sim2 = ct.ReactorNet([r2])

    step_cnt = 0
    while sim2.time < 0.002:  # 2.0 ms lean burnout
        sim2.step()
        step_cnt += 1
        if step_cnt % 3 == 0 or sim2.time >= 0.002:
            times.append(t1_end + sim2.time)
            temps.append(gas_lean.T)
            pressures.append(gas_lean.P)
            stages.append(2)
            for i, sp in enumerate(gas_lean.species_names):
                species_X[sp].append(gas_lean.X[i])

    print(f"   Stage 2 Complete: exit T = {temps[-1]:.1f} K, exit NO = {species_X['NO'][-1]*1e6:.1f} ppm")

    df = pd.DataFrame({"time": times, "T": temps, "P": pressures, "stage": stages})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals

    return df, t1_end

def main():
    print("=================================================================")
    print("  pySMART-MS: Decoupled Multi-Stage RQL Combustion Analysis      ")
    print("=================================================================\n")

    df, t_stage1_end = simulate_rql_combustor()
    print(f"\nComposite RQL trajectory generated: {len(df)} points across 2 stages.")

    print("\n4. Executing pySMART with Multi-Stage Jump Tracking (SMART-MS)...")
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        stage_col="stage",    # Enables discrete stage mass-jump tracking
        multigrid=True,
        n_workers=4
    )

    print("\n5. Identified Stage Slices:")
    for st_name, s_slice in result.stage_slices.items():
        print(f"   Stage {st_name}: rows {s_slice.start} to {s_slice.stop}")

    print("\n6. Integrated Cumulative Yields for NO:")
    df_no = result.get_cumulative_yields(target="NO")
    print(df_no.to_string(index=False))

    print("\n7. Generating publication diagnostic plots across both combustion zones...")
    t_boundary_ms = t_stage1_end * 1e3

    # 7a. Plot allocated pathway rates
    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=True)
    ax_rates.axvline(t_boundary_ms, color="gray", linestyle=":", lw=1.5, label=f"Air Quench ({t_boundary_ms:.1f} ms)")
    ax_rates.set_title("Allocated NO Source Rates across RQL Stages (SMART-MS)")
    ax_rates.legend(frameon=True)
    fig_rates.savefig("examples/02_multistage_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_rates_no.png")

    # 7b. Plot fractional formation dominance
    fig_dom, ax_dom = result.plot_dominance(target="NO", t_ms=True)
    ax_dom.axvline(t_boundary_ms, color="gray", linestyle=":", lw=1.5, label=f"Air Quench ({t_boundary_ms:.1f} ms)")
    ax_dom.set_title("NO Fractional Formation Dominance across RQL Stages")
    ax_dom.legend(frameon=True)
    fig_dom.savefig("examples/02_multistage_dominance_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_dominance_no.png")

    # 7c. Plot cumulative pools Q
    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=True)
    ax_pools.axvline(t_boundary_ms, color="gray", linestyle=":", lw=1.5, label=f"Air Quench ({t_boundary_ms:.1f} ms)")
    ax_pools.set_title("Cumulative NO Accounting Pools across RQL Stages (SMART-MS)")
    ax_pools.legend(frameon=True)
    fig_pools.savefig("examples/02_multistage_pools_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_pools_no.png")

    print("\nExample 2 finished successfully!")

if __name__ == "__main__":
    main()
