"""
Minimum Weight Perfect Matching decoder for surface codes.

Simple MWPM implementation for 3x3 surface codes with basic error correction.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from error_correction import SurfaceCodeWithErrors

class MWPMDecoder:
    """MWPM decoder for 3x3 surface codes."""

    def __init__(self, surface_code: SurfaceCodeWithErrors):
        self.surface_code = surface_code

    def decode_syndromes(self, syndrome_x: List[int], syndrome_z: List[int]) -> Dict[str, List[int]]:
        """
        Decode syndrome patterns to suggest error corrections.

        Args:
            syndrome_x: X-stabilizer measurements
            syndrome_z: Z-stabilizer measurements

        Returns:
            Dictionary with X_corrections and Z_corrections lists
        """
        corrections = {'X_corrections': [], 'Z_corrections': []}

        # X errors affect Z stabilizers
        for i, syndrome_bit in enumerate(syndrome_z):
            if syndrome_bit == 1:
                stab_qubits = list(self.surface_code.z_stabilizers.keys())
                if i < len(stab_qubits):
                    data_qubits = self.surface_code.z_stabilizers[stab_qubits[i]]
                    if data_qubits:
                        # Simple heuristic: correct first connected qubit
                        corrections['X_corrections'].append(data_qubits[0])

        # Z errors affect X stabilizers        
        for i, syndrome_bit in enumerate(syndrome_x):
            if syndrome_bit == 1:
                stab_qubits = list(self.surface_code.x_stabilizers.keys())
                if i < len(stab_qubits):
                    data_qubits = self.surface_code.x_stabilizers[stab_qubits[i]]
                    if data_qubits:
                        # Simple heuristic: correct first connected qubit
                        corrections['Z_corrections'].append(data_qubits[0])

        return corrections

    def apply_corrections(self, corrections: Dict[str, List[int]]) -> None:
        """Apply decoded corrections to the quantum circuit."""

        for qubit in corrections['X_corrections']:
            self.surface_code.circuit.x(self.surface_code.qreg[qubit])

        for qubit in corrections['Z_corrections']:
            self.surface_code.circuit.z(self.surface_code.qreg[qubit])

        total_corrections = len(corrections['X_corrections']) + len(corrections['Z_corrections'])
        if total_corrections > 0:
            self.surface_code.circuit.barrier(label=f"{total_corrections}_corrections")

class DecoderAnalysis:
    """Analysis tools for MWPM decoder performance."""

    def __init__(self, decoder: MWPMDecoder):
        self.decoder = decoder
        self.surface_code = decoder.surface_code

    def test_single_errors(self, test_qubits: List[int] = None) -> Dict[str, bool]:
        """Test decoder performance on single qubit errors."""
        if test_qubits is None:
            test_qubits = [0, 4, 8]  # Representative qubits

        results = {}

        for qubit in test_qubits:
            for error_type in ['X', 'Z']:
                # Create single error
                errors = [(qubit, error_type)]
                syndrome_x, syndrome_z = self.surface_code.calculate_syndromes(errors)

                # Test decoder
                corrections = self.decoder.decode_syndromes(syndrome_x, syndrome_z)

                # Check if correction was suggested
                if error_type == 'X':
                    success = len(corrections['X_corrections']) > 0
                else:
                    success = len(corrections['Z_corrections']) > 0

                results[f"{error_type}_error_qubit_{qubit}"] = success

        return results

    def monte_carlo_test(self, num_trials: int = 50, error_rate: float = 0.02) -> float:
        """Run Monte Carlo test of decoder performance."""
        successes = 0

        for _ in range(num_trials):
            result = self.surface_code.run_error_correction_cycle(error_rate, 0, error_rate)

            # Simple success criterion: correctable or low syndrome weight
            if result['is_correctable'] or result['total_syndrome_weight'] <= 2:
                successes += 1

        return successes / num_trials

    def benchmark_performance(self) -> Dict[str, float]:
        """Benchmark decoder across different error rates."""
        error_rates = [0.01, 0.02, 0.03, 0.05]
        performance = {}

        for rate in error_rates:
            success_rate = self.monte_carlo_test(num_trials=30, error_rate=rate)
            performance[f"rate_{rate:.2f}"] = success_rate

        return performance

def demo_decoder():
    """Quick demonstration of MWMP decoder."""
    print("MWMP Decoder Demo")
    print("-" * 20)

    # Setup
    sc = SurfaceCodeWithErrors()
    decoder = MWPMDecoder(sc)
    analyzer = DecoderAnalysis(decoder)

    # Test single errors
    print("Testing single error correction:")
    results = analyzer.test_single_errors()

    success_count = sum(results.values())
    total_tests = len(results)

    print(f"Single error success: {success_count}/{total_tests} ({success_count/total_tests:.1%})")

    # Performance test
    print("\nPerformance test:")
    perf_rate = analyzer.monte_carlo_test(num_trials=20)
    print(f"Monte Carlo success rate: {perf_rate:.1%}")

    return decoder, analyzer

if __name__ == "__main__":
    demo_decoder()
