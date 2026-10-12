import pytest
import numpy as np
import pandas as pd
import cantera as ct
import pysmart as smart

def test_pysmart_single_stage_ch4():
    """Validates pySMART on a 0D constant pressure CH4/air reactor trajectory."""
    # 1. Run a fast Cantera 0D simulation
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1200.0, 15.0 * ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    
    times = []
    temps = []
    pressures = []
    species_X = {sp: [] for sp in gas.species_names}
    
    t = 0.0
    while t < 0.005:
        sim.step()
        t = sim.time
        times.append(t)
        temps.append(gas.T)
        pressures.append(gas.P)
        for i, sp in enumerate(gas.species_names):
            species_X[sp].append(gas.X[i])
            
    df = pd.DataFrame({"time": times, "T": temps, "P": pressures})
    for sp, vals in species_X.items():
        df[f"X_{sp}"] = vals
        
    # Cap points for quick test execution
    if len(df) > 80:
        df = df.iloc[::len(df)//80].copy().reset_index(drop=True)
        
    # 2. Run pySMART analysis
    result = smart.analyze(
        df=df,
        mech="gri30.yaml",
        time_col="time",
        tracked_element="N",
        target_species="NO",
        n_workers=1,
        multigrid=False
    )
    
    # 3. Assert properties
    assert isinstance(result, smart.SMARTResult)
    assert "Thermal" in result.sub_mechs
    assert "Prompt" in result.sub_mechs
    
    df_yields = result.get_cumulative_yields(target="NO")
    assert not df_yields.empty
    assert "formation_yield" in df_yields.columns
    
    # Conservation check: sum of allocated sources should match net rate
    sp_no_idx = result._get_target_idx("NO")
    u_alloc_sum = np.sum(result.U[sp_no_idx, :, :], axis=0)
    assert len(u_alloc_sum) == len(df)
