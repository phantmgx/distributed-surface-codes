"""
Distributed Quantum Error Correction Framework

A modular implementation of surface codes for distributed quantum computing
architecture analysis.
"""


# Make imports available at package level
from .surface_code_33 import SurfaceCode33
from .error_correction import SurfaceCodeWithErrors
try:
    from .quantum_surface_code import QuantumSurfaceCode
except ImportError:
    print("Quantum simulation requires Qiskit installation")
    QuantumSurfaceCode = None
