
"""
3x3 Surface Code Implementation for Quantum Error Correction

This module implements a distance-3 surface code using Qiskit for quantum error correction research.
Author: Generated for quantum computing research
Date: October 2025
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_circuit_layout
import numpy as np
from typing import List, Dict, Tuple, Optional

class SurfaceCode33:
    """
    A 3x3 surface code implementation for quantum error correction.

    This class implements a distance-3 surface code with:
    - 9 data qubits arranged in a 3x3 grid
    - 8 stabilizer qubits (4 X-stabilizers, 4 Z-stabilizers)  
    - Methods for logical state encoding and stabilizer measurements

    Qubit Layout:
    Data qubits (D) and stabilizer qubits (X for X-stabilizers, Z for Z-stabilizers):

        D0  X0  D1  X1  D2
        Z0      Z1      Z2
        D3  X2  D4  X3  D5
        Z3      Z4      Z5
        D6  X4  D7  X5  D8

    Physical qubit mapping:
    - Data qubits: 0-8 (arranged in 3x3 grid as shown above)
    - X-stabilizers: 9-12 (X0=9, X1=10, X2=11, X3=12)
    - Z-stabilizers: 13-16 (Z0=13, Z1=14, Z2=15, Z3=16)

    The stabilizers are defined as:
    - X-stabilizers (plaquette operators): measure X⊗X⊗X⊗X on 2x2 plaquettes
    - Z-stabilizers (vertex operators): measure Z⊗Z⊗Z⊗Z around vertices
    """

    def __init__(self):
        """Initialize the 3x3 surface code with proper qubit layout and stabilizer definitions."""

        # Define the qubit layout
        self.num_data_qubits = 9
        self.num_x_stabilizers = 4  
        self.num_z_stabilizers = 4
        self.total_qubits = self.num_data_qubits + self.num_x_stabilizers + self.num_z_stabilizers

        # Data qubit indices (arranged in 3x3 grid)
        self.data_qubits = list(range(9))

        # Stabilizer qubit indices
        self.x_stabilizer_qubits = list(range(9, 13))  # qubits 9-12
        self.z_stabilizer_qubits = list(range(13, 17))  # qubits 13-16

        # Define X-stabilizer connections (plaquette operators)
        # Each X-stabilizer acts on 4 data qubits in a 2x2 plaquette
        self.x_stabilizers = {
            9: [0, 1, 3, 4],   # X0: top-left plaquette (qubits in positions (0,0), (0,1), (1,0), (1,1))
            10: [1, 2, 4, 5],  # X1: top-right plaquette (qubits in positions (0,1), (0,2), (1,1), (1,2))
            11: [3, 4, 6, 7],  # X2: bottom-left plaquette (qubits in positions (1,0), (1,1), (2,0), (2,1))
            12: [4, 5, 7, 8]   # X3: bottom-right plaquette (qubits in positions (1,1), (1,2), (2,1), (2,2))
        }

        # Define Z-stabilizer connections (vertex operators)
        # Each Z-stabilizer acts on 4 data qubits around a vertex (same connectivity as X-stabilizers)
        self.z_stabilizers = {
            13: [0, 1, 3, 4],  # Z0: top-left vertex
            14: [1, 2, 4, 5],  # Z1: top-right vertex  
            15: [3, 4, 6, 7],  # Z2: bottom-left vertex
            16: [4, 5, 7, 8]   # Z3: bottom-right vertex
        }

        # Initialize quantum registers
        self.qreg = QuantumRegister(self.total_qubits, 'q')
        self.creg_x = ClassicalRegister(self.num_x_stabilizers, 'x_syndrome')
        self.creg_z = ClassicalRegister(self.num_z_stabilizers, 'z_syndrome')
        self.creg_data = ClassicalRegister(self.num_data_qubits, 'data')

        # Initialize quantum circuit
        self.circuit = QuantumCircuit(
            self.qreg, 
            self.creg_x, 
            self.creg_z, 
            self.creg_data,
            name="SurfaceCode33"
        )

    def visualize_layout(self) -> None:
        """Print the qubit grid layout for visualization."""
        print("3x3 Surface Code Qubit Layout:")
        print("=" * 40)
        print("Grid representation (D=Data, X=X-stabilizer, Z=Z-stabilizer):")
        print()
        print("    D0   X0   D1   X1   D2")
        print("  Z0       Z1       Z2")  
        print("    D3   X2   D4   X3   D5")
        print("  Z3       Z4       Z5")
        print("    D6   X4   D7   X5   D8")
        print()
        print("Physical qubit index mapping:")
        print(f"  Data qubits:      {self.data_qubits}")
        print(f"  X-stabilizers:    {self.x_stabilizer_qubits}")
        print(f"  Z-stabilizers:    {self.z_stabilizer_qubits}")
        print(f"  Total qubits:     {self.total_qubits}")
        print()
        print("Stabilizer operator definitions:")
        print("  X-stabilizers (plaquette operators):")
        for stab_qubit, data_qubits in self.x_stabilizers.items():
            print(f"    X-stabilizer {stab_qubit}: X⊗X⊗X⊗X on data qubits {data_qubits}")
        print("  Z-stabilizers (vertex operators):")
        for stab_qubit, data_qubits in self.z_stabilizers.items():
            print(f"    Z-stabilizer {stab_qubit}: Z⊗Z⊗Z⊗Z on data qubits {data_qubits}")

    def encode_logical_zero(self) -> None:
        """
        Encode the logical |0⟩ state.

        For the surface code, the logical |0⟩ state is the +1 eigenstate of all 
        stabilizer operators. Since all qubits are initialized to |0⟩ by default,
        this state automatically satisfies all stabilizer conditions:
        - All X-stabilizers measure +1 (even parity of X-errors)
        - All Z-stabilizers measure +1 (even parity of Z-errors)
        """
        # All qubits start in |0⟩ by default, which is already the logical |0⟩ state
        # Add a barrier for circuit organization and clarity
        self.circuit.barrier(label="Logical |0⟩ encoded")

    def encode_logical_one(self) -> None:
        """
        Encode the logical |1⟩ state.

        The logical |1⟩ state is created by applying a logical X operator to the 
        logical |0⟩ state. For the surface code, a logical X operator consists of 
        a chain of physical X gates that spans the code from one boundary to the 
        opposite boundary.

        For this 3x3 implementation, we use a vertical logical X operator that 
        applies X gates to the middle column (qubits 1, 4, 7).
        """
        # Apply logical X operator: vertical chain through middle column
        logical_x_chain = [1, 4, 7]  # Middle column data qubits

        for qubit in logical_x_chain:
            self.circuit.x(self.qreg[qubit])

        # Add barrier for circuit organization
        self.circuit.barrier(label="Logical |1⟩ encoded")

    def add_stabilizer_measurements(self) -> None:
        """
        Add stabilizer measurement circuits for both X and Z stabilizers.

        This method implements the standard stabilizer measurement protocol:

        1. X-stabilizers (plaquette operators): 
           - Use ancilla qubits in |+⟩ state (apply H gate)
           - Apply CNOT gates from ancilla to each data qubit in the plaquette
           - Measure ancilla in X-basis (apply H gate before measurement)

        2. Z-stabilizers (vertex operators):
           - Use ancilla qubits in |0⟩ state  
           - Apply controlled-Z gates between data qubits and ancilla
           - Measure ancilla in Z-basis (direct measurement)

        The measurement outcomes give the syndrome information:
        - 0 (or +1): even parity, no error detected
        - 1 (or -1): odd parity, error detected in the stabilizer support
        """

        # Measure X-stabilizers (plaquette operators)
        for i, (stab_qubit, data_qubits) in enumerate(self.x_stabilizers.items()):
            # Reset ancilla qubit to |0⟩
            self.circuit.reset(self.qreg[stab_qubit])

            # Prepare ancilla in |+⟩ state for X-basis measurement
            self.circuit.h(self.qreg[stab_qubit])

            # Apply controlled-X gates from ancilla to each data qubit
            # This entangles the ancilla with the X-parity of the data qubits
            for data_qubit in data_qubits:
                self.circuit.cx(self.qreg[stab_qubit], self.qreg[data_qubit])

            # Measure ancilla in X-basis (apply H before Z-basis measurement)
            self.circuit.h(self.qreg[stab_qubit])
            self.circuit.measure(self.qreg[stab_qubit], self.creg_x[i])

        # Add barrier between X and Z stabilizer measurements
        self.circuit.barrier(label="X-stabilizers measured")

        # Measure Z-stabilizers (vertex operators)
        for i, (stab_qubit, data_qubits) in enumerate(self.z_stabilizers.items()):
            # Reset ancilla qubit to |0⟩
            self.circuit.reset(self.qreg[stab_qubit])

            # Apply controlled-Z gates from each data qubit to ancilla
            # This entangles the ancilla with the Z-parity of the data qubits
            for data_qubit in data_qubits:
                self.circuit.cz(self.qreg[data_qubit], self.qreg[stab_qubit])

            # Measure ancilla directly in Z-basis
            self.circuit.measure(self.qreg[stab_qubit], self.creg_z[i])

        # Final barrier after all stabilizer measurements
        self.circuit.barrier(label="Z-stabilizers measured")

    def measure_data_qubits(self) -> None:
        """
        Measure all data qubits in the computational basis.

        This is typically done at the end of an error correction cycle
        to read out the final logical state or for debugging purposes.
        """
        for i, data_qubit in enumerate(self.data_qubits):
            self.circuit.measure(self.qreg[data_qubit], self.creg_data[i])

        self.circuit.barrier(label="Data qubits measured")

    def get_circuit(self) -> QuantumCircuit:
        """
        Return the current quantum circuit.

        Returns:
            QuantumCircuit: The surface code quantum circuit
        """
        return self.circuit

    def reset_circuit(self) -> None:
        """Reset the circuit to initial state (empty circuit)."""
        self.circuit = QuantumCircuit(
            self.qreg, 
            self.creg_x, 
            self.creg_z, 
            self.creg_data,
            name="SurfaceCode33"
        )

    def get_circuit_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current circuit.

        Returns:
            Dict containing circuit depth, width, and gate counts
        """
        return {
            'depth': self.circuit.depth(),
            'width': self.circuit.width(),
            'size': self.circuit.size(),
            'num_qubits': self.circuit.num_qubits,
            'num_clbits': self.circuit.num_clbits
        }


def create_example_circuits() -> Tuple[SurfaceCode33, QuantumCircuit, QuantumCircuit]:
    """
    Create example circuits showing logical |0⟩ and |1⟩ encoding with stabilizer measurements.

    Returns:
        Tuple of (SurfaceCode33 instance, logical_0_circuit, logical_1_circuit)
    """

    # Create surface code instance
    surface_code = SurfaceCode33()

    # Example 1: Logical |0⟩ encoding
    surface_code.reset_circuit()
    surface_code.encode_logical_zero()
    surface_code.add_stabilizer_measurements()
    logical_0_circuit = surface_code.get_circuit().copy()

    # Example 2: Logical |1⟩ encoding  
    surface_code.reset_circuit()
    surface_code.encode_logical_one()
    surface_code.add_stabilizer_measurements()
    logical_1_circuit = surface_code.get_circuit().copy()

    return surface_code, logical_0_circuit, logical_1_circuit


if __name__ == "__main__":
    """Demo script showing the SurfaceCode33 class functionality."""

    print("=" * 50)
    print("3x3 Surface Code Implementation Demo")
    print("=" * 50)

    # Create and demonstrate the surface code
    surface_code, circuit_0, circuit_1 = create_example_circuits()

    # Show the qubit layout
    surface_code.visualize_layout()

    print("\n" + "=" * 50)
    print("Circuit Statistics")
    print("=" * 50)

    # Reset to get fresh stats for logical |0⟩
    surface_code.reset_circuit()
    surface_code.encode_logical_zero()
    surface_code.add_stabilizer_measurements()
    stats_0 = surface_code.get_circuit_stats()

    print(f"Logical |0⟩ circuit:")
    for key, value in stats_0.items():
        print(f"  {key}: {value}")

    # Reset to get fresh stats for logical |1⟩  
    surface_code.reset_circuit()
    surface_code.encode_logical_one()
    surface_code.add_stabilizer_measurements()
    stats_1 = surface_code.get_circuit_stats()

    print(f"\nLogical |1⟩ circuit:")
    for key, value in stats_1.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Implementation Complete!")
    print("Next: Use this foundation for error injection and syndrome analysis")
    print("=" * 50)
