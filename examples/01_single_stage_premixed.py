"""
Example 1: Single-Stage Premixed Reactor Pathway Analysis
Mirrors Thesis_Q1_Q2/Reaction_Pathway_Analysis/SMART_Algorithm/example.py:
- Simulating stoichiometric CH4/air ignition (Tin=600 K, P=5 atm, GRI-Mech 3.0)
- Running pySMART to track Nitrogen (N) pathways forming NO
- Resolves Prompt NO peak at the ignition flame front followed by Thermal NO post-flame rise
- Generates publication diagnostic plots: Rates, Dominance stackplot, Phase portrait, and Network graph
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def main():
    print("1. Simulating CH4/air ignition trajectory matching Thesis_Q1_Q2 example (600 K, 5 atm)...")
    mech = "gri30.yaml"
    gas = ct.Solution(mech)
    gas.TP = 600.0, 5.0 * ct.one_atm
    gas.set_equivalence_ratio(1.0, "CH4:1", "O2:1.0, N2:3.76", basis="mole")

    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])

    states = ct.SolutionArray(gas, extra=["time"])
    t_val = 0.0
    for n in range(60000):
        old_t_val = t_val
        t_val = sim.step()
        dt = t_val - old_t_val
        if dt > 0.01 and gas.T > 1000.0:
            break
        if n % 5 == 0:
            states.append(r.thermo.state, time=t_val)

    df = states.to_pandas(species="X")
    df["P"] = 5.0 * ct.one_atm

    print(f"   Trajectory generated: {len(df)} states across auto-ignition.")
    print(f"   Temperature span: {df['T'].min():.1f} K -> {df['T'].max():.1f} K.")

    print("\n2. Executing pySMART Reaction Pathway Analysis...")
    result = smart.analyze(
        df=df,
        mech=mech,
        time_col="time",
        tracked_element="N",
        target_species="NO",
        fallback_pressure_pa=5.0 * ct.one_atm,
        multigrid=True,
        n_workers=8
    )

    print("\n3. Integrated Cumulative NO Yields:")
    df_yields = result.get_cumulative_yields(target="NO")
    print(df_yields.to_string(index=False))

    print("\n4. Generating publication diagnostic plots...")
    
    # 4a. Time-resolved allocated rates (U matrix)
    # The flame reaction occurs over a sharp window, so we center x-limits around the active ignition zone
    t_ignition = df.loc[df["T"] > 1000.0, "time"].iloc[0]
    dt_window = 0.5  # 500 ms window covering induction through post-flame
    t_min = max(0.0, t_ignition - 0.2)
    t_max = t_ignition + 0.3

    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=False)
    ax_rates.set_xlim(t_min, t_max)
    ax_rates.set_title("Allocated NO Source Rate (U Matrix)")
    fig_rates.savefig("examples/01_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_rates_no.png")

    # 4b. Fractional formation dominance stackplot
    fig_dom, ax_dom = result.plot_dominance(target="NO", t_ms=False)
    ax_dom.set_xlim(t_min, t_max)
    ax_dom.set_title("NO Fractional Formation Dominance")
    fig_dom.savefig("examples/01_dominance_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_dominance_no.png")

    # 4c. Phase portrait (ROP vs. Temperature) - Intuitive state-space view
    fig_phase, ax_phase = result.plot_phase_portrait(df=df, x_col="T", target="NO")
    ax_phase.set_title("1D Phase Portrait (NO ROP vs Temperature)")
    fig_phase.savefig("examples/01_phase_portrait_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_phase_portrait_no.png")

    # 4d. Cumulative accounting pools Q
    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=False)
    ax_pools.set_xlim(t_min, t_max)
    ax_pools.set_title("Cumulative NO Accounting Pools (Q Matrix)")
    fig_pools.savefig("examples/01_pools_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_pools_no.png")

    # 4e. Reaction Pathway Network Graph
    fig_net, ax_net, pos = result.plot_network(
        df=df,
        target="NO",
        threshold=0.03
    )
    ax_net.set_title("Integrated Reaction Pathway Diagram")
    fig_net.savefig("examples/01_pathway_network_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_pathway_network_no.png")

    print("\nExample 1 finished successfully!")

if __name__ == "__main__":
    main()
