"""Model zoo for the 3D NIST PINN (SPEC item 6)."""
from .fcnn import FCNN, INPUT_ORDER, OUTPUT_ORDER, Swish, init_weights

__all__ = ["FCNN", "Swish", "init_weights", "INPUT_ORDER", "OUTPUT_ORDER"]
