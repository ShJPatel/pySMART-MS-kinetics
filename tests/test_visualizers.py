"""
test_visualizers.py

Tests all publication plotting helpers in pySMART:
- .plot_rates()
- .plot_dominance()
- .plot_pools()
- .plot_network()
- .plot_flux_budget()
- .plot_phase_portrait()
"""

import pytest
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import pysmart as smart

@pytest.fixture(scope="module")
def sample_result_and_df():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, 15.0 * ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])

    times, temps, pressures = [], [], []
    species_X = {sp: [] for sp in gas.species_names}

    while sim.time < 0.002:
        sim.step()
        times.append(sim.time)
        temps.append(gas.T)
        pressures.append(gas.P)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])

    df = pd.DataFrame({"time": times, "T": temps, "P": pressures})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals

    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        n_workers=1,
        multigrid=False
    )
    return result, df

def test_plot_rates(sample_result_and_df):
    result, _ = sample_result_and_df
    fig, ax = result.plot_rates(target="NO")
    assert fig is not None
    plt.close(fig)

def test_plot_dominance(sample_result_and_df):
    result, _ = sample_result_and_df
    fig, ax = result.plot_dominance(target="NO")
    assert fig is not None
    plt.close(fig)

def test_plot_pools(sample_result_and_df):
    result, _ = sample_result_and_df
    fig, ax = result.plot_pools(target="NO")
    assert fig is not None
    plt.close(fig)

def test_plot_flux_budget(sample_result_and_df):
    result, _ = sample_result_and_df
    fig, (ax_top, ax_bot) = result.plot_flux_budget(target="NO")
    assert fig is not None
    plt.close(fig)

def test_plot_phase_portrait(sample_result_and_df):
    result, df = sample_result_and_df
    fig, ax = result.plot_phase_portrait(df=df, x_col="T", target="NO")
    assert fig is not None
    plt.close(fig)

def test_plot_network(sample_result_and_df):
    result, df = sample_result_and_df
    fig, ax, pos = result.plot_network(df=df, threshold=0.05, target="NO")
    assert fig is not None
    assert isinstance(pos, dict)
    plt.close(fig)
