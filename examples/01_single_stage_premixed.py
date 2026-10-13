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

    # Zero-shift time relative to ignition onset (T >= 1000 K) so flame dynamics are centered
    t_ignition = df.loc[df["T"] >= 1000.0, "time"].iloc[0]
    df["time"] = df["time"] - t_ignition

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

    print("\n4. Generating publication diagnostic plots zoomed in on flame zone...")
    
    # Flame zone window: Prompt NO spikes at ~2.05 ms and Thermal NO rises at ~2.08 ms
    # Zooming in tightly to [1.8, 2.6] ms clearly exposes the flame front radical dynamics and prompt peak!
    flame_xlim = (1.8, 2.6)
    # Pathway rates peak around 0.02 - 0.03 kmol/(m³·s). Clipping the extreme net rate peak gives extreme resolution on the pathways!
    flame_ylim = (-0.001, 0.035)

    # 4a. Time-resolved allocated rates (U matrix)
    fig_rates, ax_rates = result.plot_rates(target="NO", t_ms=True)
    ax_rates.set_xlim(flame_xlim)
    ax_rates.set_ylim(flame_ylim)
    ax_rates.set_title("Allocated NO Source Rate in Flame Zone (U Matrix)")
    fig_rates.savefig("examples/01_rates_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_rates_no.png")

    # 4b. Fractional formation dominance stackplot
    fig_dom, ax_dom = result.plot_dominance(target="NO", t_ms=True)
    ax_dom.set_xlim(flame_xlim)
    ax_dom.set_title("NO Fractional Formation Dominance in Flame Zone")
    fig_dom.savefig("examples/01_dominance_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_dominance_no.png")

    # 4c. Phase portrait (ROP vs. Temperature) - Intuitive state-space view
    fig_phase, ax_phase = result.plot_phase_portrait(df=df, x_col="T", target="NO")
    ax_phase.set_title("1D Phase Portrait (NO ROP vs Temperature)")
    fig_phase.savefig("examples/01_phase_portrait_no.png", dpi=300, bbox_inches="tight")
    print("   Saved: examples/01_phase_portrait_no.png")

    # 4d. Cumulative accounting pools Q
    fig_pools, ax_pools = result.plot_pools(target="NO", t_ms=True)
    ax_pools.set_xlim(flame_xlim)
    ax_pools.set_title("Cumulative NO Pools in Flame Zone (Q Matrix)")
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
