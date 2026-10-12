"""
pysmart: Sub-Mechanism Accounting and Reaction Tracing in Python
"""

from .api import analyze
from .results.container import SMARTResult
from .mechanisms.sub_mechanisms import SubMechanism, DEFAULT_PALETTE
from .mechanisms.registry import get_mechanism_mapping, auto_classify_mechanism

__version__ = "0.1.0"
__all__ = [
    "analyze",
    "SMARTResult",
    "SubMechanism",
    "DEFAULT_PALETTE",
    "get_mechanism_mapping",
    "auto_classify_mechanism"
]
