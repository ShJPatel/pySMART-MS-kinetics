"""
Example 1: Single-Stage Premixed Reactor Pathway Analysis
Demonstrates basic pySMART workflow on a 0D Cantera reactor trajectory:
- Simulating a stoichiometric CH4/air ignition trajectory
- Running pySMART to track Nitrogen (N) pathways forming NO
- Extracting cumulative yields and plotting pathway diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def main():
    print("1. Simulating CH4/air constant pressure ignition trajectory using Cantera...")
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1100.0, 15.0 * ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])

    times, temps, pressures = [], [], []
    species_X = {sp: [] for sp in gas.species_names}

    t = 0.0
    while t < 0.003:
        sim.step()
        t = sim.time
        times.append(t)
        temps.append(gas.T)
        pressures.append(gas.P)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])

    # Construct standard trajectory DataFrame
    df = pd.DataFrame({"time": times, "T": temps, "P": pressures})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals

    print(f"   Trajectory generated: {len(df)} adaptive timesteps.")

    print("\n2. Executing pySMART Reaction Pathway Analysis...")
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        multigrid=True,
        n_workers=4
    )

    print("\n3. Integrated Cumulative NO Yields:")
    df_yields = result.get_cumulative_yields(target="NO")
    print(df_yields.to_string(index=False))

    print("\n4. Generating publication diagnostic plots...")
    # Plot allocated pathway rates
    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=True)
    fig_rates.savefig("examples/01_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_rates_no.png")

    # Plot fractional dominance stackplot
    fig_dom, ax_dom = result.plot_dominance(target="NO", t_ms=True)
    fig_dom.savefig("examples/01_dominance_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_dominance_no.png")

    # Plot cumulative pools Q
    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=True)
    fig_pools.savefig("examples/01_pools_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_pools_no.png")

    print("\nExample 1 finished successfully!")

if __name__ == "__main__":
    main()
