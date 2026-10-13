"""
Example 2: Multi-Stage Staged Combustor Analysis (SMART-MS)
Demonstrates pySMART on an axially staged combustor:
- Stage 1: Stoichiometric primary flame products
- Stage 2: Secondary fuel injection (rich relaxation) with discrete mixing jump tracking
- Resolves prompt DeNOx destruction and exact mass closure across stage transitions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def main():
    print("1. Simulating two-stage combustion trajectory (Primary Flame -> Rich Relaxation)...")
    gas = ct.Solution("gri30.yaml")

    # Stage 1: Stoichiometric CH4/air flame to equilibrium
    gas.TPX = 1300.0, 15.0 * ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
    r1 = ct.IdealGasConstPressureReactor(gas)
    sim1 = ct.ReactorNet([r1])

    times, temps, pressures, stages = [], [], [], []
    species_X = {sp: [] for sp in gas.species_names}

    while sim1.time < 0.002:
        sim1.step()
        times.append(sim1.time)
        temps.append(gas.T)
        pressures.append(gas.P)
        stages.append(1)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])

    t_stage1_end = times[-1]

    # Stage 2: Secondary fuel injection (simulate rich staging)
    # Gas mixture is diluted with secondary CH4
    gas.TPX = 1450.0, 15.0 * ct.one_atm, "CH4:0.8, O2:0.5, N2:6.0"
    r2 = ct.IdealGasConstPressureReactor(gas)
    sim2 = ct.ReactorNet([r2])

    while sim2.time < 0.002:
        sim2.step()
        times.append(t_stage1_end + sim2.time)
        temps.append(gas.T)
        pressures.append(gas.P)
        stages.append(2)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])

    df = pd.DataFrame({"time": times, "T": temps, "P": pressures, "stage": stages})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals

    print(f"   Trajectory generated: {len(df)} points across 2 stages.")

    print("\n2. Executing pySMART with Multi-Stage Jump Tracking (SMART-MS)...")
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        stage_col="stage",    # Enables SMART-MS jump tracking
        multigrid=True,
        n_workers=4
    )

    print("\n3. Identified Stage Slices:")
    for st_name, s_slice in result.stage_slices.items():
        print(f"   Stage {st_name}: rows {s_slice.start} to {s_slice.stop}")

    print("\n4. Integrated Cumulative Yields Across Both Stages:")
    df_yields = result.get_cumulative_yields(target="NO")
    print(df_yields.to_string(index=False))

    print("\n5. Generating diagnostic plots across stage boundary...")
    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=True)
    ax_rates.axvline(t_stage1_end * 1e3, color="gray", linestyle=":", label="Stage Boundary")
    ax_rates.legend(frameon=True)
    fig_rates.savefig("examples/02_multistage_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_rates_no.png")

    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=True)
    ax_pools.axvline(t_stage1_end * 1e3, color="gray", linestyle=":", label="Stage Boundary")
    ax_pools.legend(frameon=True)
    fig_pools.savefig("examples/02_multistage_pools_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/02_multistage_pools_no.png")

    print("\nExample 2 finished successfully!")

if __name__ == "__main__":
    main()
