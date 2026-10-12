from enum import Enum

class SubMechanism(Enum):
    """Standard sub-mechanism names for chemical pathway attribution."""
    THERMAL = "Thermal"
    PROMPT = "Prompt"
    N2O = "N2O"
    NNH = "NNH"
    NO2 = "NO2"
    HNO = "HNO"
    REBURN = "Reburn"
    HCN_OX = "HCN Oxidation"
    HNCO_OX = "HNCO Oxidation"
    NH3 = "NH3"
    DENOX = "DeNOx"
    STAGED_MIXING = "Staged Mixing"
    OTHER_FORMATION = "Other Formation"
    OTHER_DEPLETION = "Other Depletion"

# High-contrast accessible color scheme with user-override support
DEFAULT_PALETTE = {
    SubMechanism.THERMAL.value: '#2E7D32',       # Green
    SubMechanism.PROMPT.value: '#FFC107',        # Gold / Amber
    SubMechanism.N2O.value: '#3F51B5',           # Indigo
    SubMechanism.NNH.value: '#0097A7',           # Cyan
    SubMechanism.NO2.value: '#C62828',           # Red
    SubMechanism.HNO.value: '#AD1457',           # Crimson / Pink
    SubMechanism.REBURN.value: '#FF5722',        # Deep Orange
    SubMechanism.HCN_OX.value: '#D2691E',        # Chocolate
    SubMechanism.HNCO_OX.value: '#A0522D',       # Sienna
    SubMechanism.NH3.value: '#8D6E63',           # Warm Caramel / Tan Brown
    SubMechanism.DENOX.value: '#3E2723',         # Deep Espresso / Dark Brown
    SubMechanism.STAGED_MIXING.value: '#00BCD4', # Vivid Cyan
    "Ambiguous Pathways": '#F48FB1',
    SubMechanism.OTHER_FORMATION.value: '#757575',
    SubMechanism.OTHER_DEPLETION.value: '#BDBDBD'
}
