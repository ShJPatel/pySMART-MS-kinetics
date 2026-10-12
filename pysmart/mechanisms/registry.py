import json
import os
from typing import Dict, List, Optional
import cantera as ct
from .sub_mechanisms import SubMechanism

_CURATED_MAPPINGS = {}

def get_mechanism_mapping(mech_name: str, gas: Optional[ct.Solution] = None, version: int = 2) -> Dict[str, List[int]]:
    """
    Retrieves or generates 0-based signed reaction index mappings for SMART initiation tracking.
    
    Returns:
        dict: sub-mechanism name -> list of 0-based signed reaction indices.
    """
    if "gri30" in mech_name.lower():
        return _get_gri30_mapping(initiation_only=True)
    elif "glarborg" in mech_name.lower():
        json_path = os.path.join(os.path.dirname(__file__), "glarborg_2018_v2.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                raw = json.load(f)
            # Filter and convert to 0-based signed indices
            return _filter_and_convert_0based(raw, gas or ct.Solution(mech_name), initiation_only=True)
        
    # Auto-classify dynamically for other mechanisms if gas object is available
    if gas is None:
        ct.suppress_thermo_warnings()
        gas = ct.Solution(mech_name)
    return auto_classify_mechanism(gas)

def _convert_to_0_based(val):
    if val > 0:
        return val - 1
    elif val < 0:
        converted = abs(val) - 1
        return -float(converted) if converted == 0 else -converted
    return 0

def _get_gri30_mapping(initiation_only: bool = True) -> Dict[str, List[int]]:
    raw_1based = {
        SubMechanism.PROMPT.value: [-238, 239, 240, 241, 242],
        SubMechanism.THERMAL.value: [-178],
        SubMechanism.N2O.value: [-181, -183],
        SubMechanism.NNH.value: [-204, -205],
        SubMechanism.NO2.value: [186, -188],
        SubMechanism.HNO.value: [212],
        SubMechanism.REBURN.value: [228, 229, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
    }
    return {k: [_convert_to_0_based(r) for r in v] for k, v in raw_1based.items()}

def _filter_and_convert_0based(raw_mapping: Dict[str, list], gas: ct.Solution, initiation_only: bool = True) -> Dict[str, list]:
    filtered = {}
    for cat, rxns in raw_mapping.items():
        filtered_rxns = []
        for r_signed in rxns:
            r_idx = abs(int(r_signed)) - 1
            if r_idx < gas.n_reactions:
                rxn = gas.reactions()[r_idx]
                r_set = set(rxn.reactants.keys())
                p_set = set(rxn.products.keys())

                if cat == SubMechanism.THERMAL.value:
                    if (r_set == {'O', 'N2'} and p_set == {'N', 'NO'}) or (p_set == {'O', 'N2'} and r_set == {'N', 'NO'}):
                        filtered_rxns.append(_convert_to_0_based(r_signed))
                elif cat == SubMechanism.PROMPT.value:
                    prompt_sp = {'CH', 'CH2', 'CH2(S)', 'C', 'NCN'}
                    if ('N2' in r_set and any(c in r_set for c in prompt_sp)) or ('N2' in p_set and any(c in p_set for c in prompt_sp)):
                        filtered_rxns.append(_convert_to_0_based(r_signed))
                elif cat == SubMechanism.N2O.value:
                    if ('N2' in r_set and 'N2O' in p_set) or ('N2' in p_set and 'N2O' in r_set):
                        filtered_rxns.append(_convert_to_0_based(r_signed))
                elif cat == SubMechanism.NNH.value:
                    if ('N2' in r_set and 'NNH' in p_set) or ('N2' in p_set and 'NNH' in r_set):
                        filtered_rxns.append(_convert_to_0_based(r_signed))
                else:
                    filtered_rxns.append(_convert_to_0_based(r_signed))
        filtered[cat] = filtered_rxns
    return filtered

def auto_classify_mechanism(gas: ct.Solution) -> Dict[str, List[int]]:
    """Heuristic rule-based gateway initiation classifier for any arbitrary Cantera mechanism."""
    mapping = {
        SubMechanism.THERMAL.value: [],
        SubMechanism.PROMPT.value: [],
        SubMechanism.N2O.value: [],
        SubMechanism.NNH.value: [],
        SubMechanism.NO2.value: [],
        SubMechanism.HNO.value: [],
        SubMechanism.REBURN.value: [],
    }
    for i, rxn in enumerate(gas.reactions()):
        r_set = set(rxn.reactants.keys())
        p_set = set(rxn.products.keys())
        # Thermal: N + NO <=> N2 + O, N + O2 <=> NO + O
        if (r_set == {'O', 'N2'} and p_set == {'N', 'NO'}):
            mapping[SubMechanism.THERMAL.value].append(i)
        elif (p_set == {'O', 'N2'} and r_set == {'N', 'NO'}):
            mapping[SubMechanism.THERMAL.value].append(-float(i) if i == 0 else -i)
        # Prompt: CH/CH2/C + N2
        elif 'N2' in r_set and any(c in r_set for c in ['CH', 'CH2', 'C', 'NCN']):
            mapping[SubMechanism.PROMPT.value].append(i)
        elif 'N2' in p_set and any(c in p_set for c in ['CH', 'CH2', 'C', 'NCN']):
            mapping[SubMechanism.PROMPT.value].append(-float(i) if i == 0 else -i)
        # N2O
        elif ('N2' in r_set and 'N2O' in p_set):
            mapping[SubMechanism.N2O.value].append(i)
        elif ('N2' in p_set and 'N2O' in r_set):
            mapping[SubMechanism.N2O.value].append(-float(i) if i == 0 else -i)
        # NNH
        elif ('N2' in r_set and 'NNH' in p_set):
            mapping[SubMechanism.NNH.value].append(i)
        elif ('N2' in p_set and 'NNH' in r_set):
            mapping[SubMechanism.NNH.value].append(-float(i) if i == 0 else -i)
    return mapping
