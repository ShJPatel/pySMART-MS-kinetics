import pytest
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def test_pysmart_multistage_ms():
    """Tests multi-stage SMART-MS jump tracking and pool scaling across two simulated stages."""
    gas = ct.Solution("gri30.yaml")
    
    # Stage 1: Premixed flame to post-flame
    gas.TPX = 1400.0, 15.0 * ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
    r1 = ct.IdealGasConstPressureReactor(gas)
    sim1 = ct.ReactorNet([r1])
    
    times, temps, pressures, stages = [], [], [], []
    species_X = {sp: [] for sp in gas.species_names}
    
    for _ in range(40):
        sim1.step()
        times.append(sim1.time)
        temps.append(gas.T)
        pressures.append(gas.P)
        stages.append(1)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])
            
    # Transition to Stage 2: simulate secondary injection / dilution
    t_end_1 = times[-1]
    # Dilute gases by adding secondary stream
    gas.TPX = 1500.0, 15.0 * ct.one_atm, "CH4:0.5, N2:5.0, O2:1.0"
    r2 = ct.IdealGasConstPressureReactor(gas)
    sim2 = ct.ReactorNet([r2])
    
    for _ in range(40):
        sim2.step()
        times.append(t_end_1 + sim2.time)
        temps.append(gas.T)
        pressures.append(gas.P)
        stages.append(2)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])
            
    df = pd.DataFrame({"time": times, "T": temps, "P": pressures, "stage": stages})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals
        
    # Run SMART with multi-stage support
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        stage_col="stage",
        n_workers=1,
        multigrid=False
    )
    
    assert isinstance(result, smart.SMARTResult)
    assert len(result.stage_slices) == 2
    assert "1" in result.stage_slices
    assert "2" in result.stage_slices
