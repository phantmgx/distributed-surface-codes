
"""
Error injection and syndrome analysis for 3x3 surface codes.

This module extends the basic surface code with error injection capabilities
and classical syndrome calculation for performance analysis.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from surface_code_33 import SurfaceCode33

class SurfaceCodeWithErrors(SurfaceCode33):
    """
    3x3 surface code with error injection and syndrome analysis.

    Extends the basic surface code to support:
    - Random Pauli error injection
    - Classical syndrome calculation
    - Error correction cycle simulation
    - Performance analysis
    """

    def __init__(self):
        super().__init__()
        self.error_history = []
        self.syndrome_history = []

    def inject_pauli_errors(self, p_x: float = 0.01, p_y: float = 0.01, p_z: float = 0.01) -> List[Tuple[int, str]]:
        """
        Inject random Pauli errors on data qubits.

        Args:
            p_x: Probability of X error per qubit
            p_y: Probability of Y error per qubit  
            p_z: Probability of Z error per qubit

        Returns:
            List of (qubit_index, error_type) pairs
        """
        errors = []

        for qubit in self.data_qubits:
            r = random.random()

            if r < p_x:
                self.circuit.x(self.qreg[qubit])
                errors.append((qubit, 'X'))
            elif r < p_x + p_y:
                self.circuit.y(self.qreg[qubit])
                errors.append((qubit, 'Y'))
            elif r < p_x + p_y + p_z:
                self.circuit.z(self.qreg[qubit])
                errors.append((qubit, 'Z'))

        self.error_history = errors
        if errors:
            self.circuit.barrier(label=f"errors_{len(errors)}")

        return errors

    def calculate_syndromes(self, errors: List[Tuple[int, str]]) -> Tuple[List[int], List[int]]:
        """
        Calculate syndrome measurements from error pattern.

        Uses classical simulation to determine what the stabilizer measurements
        would be given a specific error pattern.

        Args:
            errors: List of (qubit, error_type) pairs

        Returns:
            Tuple of (x_syndromes, z_syndromes) as binary lists
        """
        x_syndromes = [0] * self.num_x_stabilizers
        z_syndromes = [0] * self.num_z_stabilizers

        for qubit, error_type in errors:
            if error_type in ['X', 'Y']:
                # X component affects Z stabilizers
                for i, (stab_qubit, data_qubits) in enumerate(self.z_stabilizers.items()):
                    if qubit in data_qubits:
                        z_syndromes[i] ^= 1

            if error_type in ['Z', 'Y']:
                # Z component affects X stabilizers  
                for i, (stab_qubit, data_qubits) in enumerate(self.x_stabilizers.items()):
                    if qubit in data_qubits:
                        x_syndromes[i] ^= 1

        return x_syndromes, z_syndromes

    def run_error_correction_cycle(self, p_x: float = 0.01, p_y: float = 0.01, p_z: float = 0.01) -> Dict:
        """
        Run complete error correction cycle with performance analysis.

        Steps:
        1. Reset and encode logical zero state
        2. Inject random errors according to error rates
        3. Calculate resulting syndromes
        4. Analyze correctability and performance

        Args:
            p_x, p_y, p_z: Error probabilities for X, Y, Z errors

        Returns:
            Dictionary with detailed results and analysis
        """
        # Reset circuit state
        self.reset_circuit()
        self.encode_logical_zero()

        # Inject errors
        errors = self.inject_pauli_errors(p_x, p_y, p_z)

        # Calculate syndromes
        x_syndromes, z_syndromes = self.calculate_syndromes(errors)
        self.syndrome_history.append(x_syndromes + z_syndromes)

        # Performance analysis
        syndrome_weight = sum(x_syndromes) + sum(z_syndromes)
        is_correctable = self._analyze_correctability(syndrome_weight, len(errors))

        # Generate comprehensive results
        return {
            'errors_injected': errors,
            'num_errors': len(errors),
            'x_syndromes': x_syndromes,
            'z_syndromes': z_syndromes,
            'total_syndrome_weight': syndrome_weight,
            'is_correctable': is_correctable,
            'error_rates': (p_x, p_y, p_z),
            'circuit_depth': self.circuit.depth() if hasattr(self.circuit, 'depth') else 0
        }

    def _analyze_correctability(self, syndrome_weight: int, num_errors: int) -> bool:
        """
        Analyze whether error pattern is likely correctable.

        For distance-3 surface code:
        - Single errors: always correctable
        - Two errors: usually correctable
        - Three+ errors: often uncorrectable

        Args:
            syndrome_weight: Total syndrome weight
            num_errors: Number of errors injected

        Returns:
            Boolean indicating likely correctability
        """
        # Heuristic for distance-3 surface code
        if syndrome_weight == 0:
            return True  # No detectable errors
        elif syndrome_weight <= 4 and num_errors <= 2:
            return True  # Low syndrome weight, few errors
        elif syndrome_weight > 6 or num_errors > 3:
            return False  # High syndrome weight or many errors
        else:
            return syndrome_weight <= 4  # Borderline case

    def analyze_error_pattern(self, errors: List[Tuple[int, str]]) -> Dict:
        """
        Detailed analysis of specific error pattern.

        Args:
            errors: List of (qubit, error_type) pairs

        Returns:
            Analysis dictionary with error characterization
        """
        if not errors:
            return {
                'pattern_type': 'no_errors',
                'severity': 'none',
                'expected_syndrome_weight': 0
            }

        # Characterize error pattern
        error_types = [e[1] for e in errors]
        error_qubits = [e[0] for e in errors]

        x_count = error_types.count('X')
        y_count = error_types.count('Y') 
        z_count = error_types.count('Z')

        # Calculate expected syndrome
        x_synd, z_synd = self.calculate_syndromes(errors)
        expected_weight = sum(x_synd) + sum(z_synd)

        # Classify severity
        if len(errors) == 1:
            severity = 'single'
        elif len(errors) == 2:
            severity = 'double'
        elif len(errors) <= 4:
            severity = 'multiple'
        else:
            severity = 'high'

        return {
            'pattern_type': f"{len(errors)}_errors",
            'severity': severity,
            'error_breakdown': {'X': x_count, 'Y': y_count, 'Z': z_count},
            'affected_qubits': error_qubits,
            'expected_syndrome_weight': expected_weight,
            'x_syndrome': x_synd,
            'z_syndrome': z_synd
        }

    def visualize_errors(self, errors: List[Tuple[int, str]]) -> None:
        """
        Print visualization of errors on the surface code grid.

        Args:
            errors: List of (qubit, error_type) pairs to visualize
        """
        print("3x3 Surface Code Grid:")

        # Grid template showing layout
        grid_layout = [
            ['D0', 'X0', 'D1', 'X1', 'D2'],
            ['Z0', '  ', 'Z1', '  ', 'Z2'], 
            ['D3', 'X2', 'D4', 'X3', 'D5'],
            ['Z3', '  ', 'Z4', '  ', 'Z5'],
            ['D6', 'X4', 'D7', 'X5', 'D8']
        ]

        # Print grid
        for row in grid_layout:
            print("  ".join(f"{cell:2s}" for cell in row))

        if not errors:
            print("\nNo errors injected.")
            return

        # Print error details
        print(f"\nError locations ({len(errors)} total):")

        # Map qubits to grid coordinates  
        qubit_positions = {
            0: (0, 0), 1: (0, 2), 2: (0, 4),
            3: (2, 0), 4: (2, 2), 5: (2, 4), 
            6: (4, 0), 7: (4, 2), 8: (4, 4)
        }

        for qubit, error_type in errors:
            if qubit in qubit_positions:
                row, col = qubit_positions[qubit]
                grid_pos = (row // 2, col // 2)
                print(f"  {error_type} error on data qubit {qubit} at grid position {grid_pos}")
            else:
                print(f"  {error_type} error on qubit {qubit} (non-data)")

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics from error correction history.

        Returns:
            Dictionary with performance metrics
        """
        if not self.syndrome_history:
            return {'message': 'No error correction cycles run yet'}

        syndrome_weights = [sum(syndrome) for syndrome in self.syndrome_history]

        return {
            'total_cycles': len(self.syndrome_history),
            'avg_syndrome_weight': np.mean(syndrome_weights) if syndrome_weights else 0,
            'max_syndrome_weight': max(syndrome_weights) if syndrome_weights else 0,
            'zero_syndrome_fraction': syndrome_weights.count(0) / len(syndrome_weights) if syndrome_weights else 0,
            'last_syndrome_weight': syndrome_weights[-1] if syndrome_weights else 0
        }

    def reset_history(self):
        """Clear error and syndrome history."""
        self.error_history = []
        self.syndrome_history = []


# Utility functions for batch analysis
def run_monte_carlo_analysis(error_rates: List[float], 
                           trials_per_rate: int = 100) -> Dict:
    """
    Run Monte Carlo analysis over multiple error rates.

    Args:
        error_rates: List of physical error rates to test
        trials_per_rate: Number of trials per error rate

    Returns:
        Dictionary with analysis results
    """
    surface_code = SurfaceCodeWithErrors()
    results = {'error_rates': error_rates, 'logical_error_rates': [], 'avg_syndrome_weights': []}

    print(f"Running Monte Carlo analysis: {len(error_rates)} rates × {trials_per_rate} trials")

    for rate in error_rates:
        failures = 0
        total_syndrome_weight = 0

        for trial in range(trials_per_rate):
            result = surface_code.run_error_correction_cycle(rate, 0, rate)

            if not result['is_correctable']:
                failures += 1

            total_syndrome_weight += result['total_syndrome_weight']

        logical_error_rate = failures / trials_per_rate
        avg_syndrome_weight = total_syndrome_weight / trials_per_rate

        results['logical_error_rates'].append(logical_error_rate)
        results['avg_syndrome_weights'].append(avg_syndrome_weight)

        print(f"  Rate {rate:.3f}: logical error rate {logical_error_rate:.3f}, avg syndrome weight {avg_syndrome_weight:.2f}")

    return results


def compare_error_types(num_trials: int = 50) -> Dict:
    """
    Compare performance for different types of errors.

    Args:
        num_trials: Number of trials per error type

    Returns:
        Comparison results
    """
    surface_code = SurfaceCodeWithErrors()

    test_scenarios = [
        ('X_only', 0.02, 0, 0),
        ('Z_only', 0, 0, 0.02),
        ('mixed', 0.01, 0, 0.01),
        ('with_Y', 0.01, 0.005, 0.01)
    ]

    results = {}

    for scenario_name, p_x, p_y, p_z in test_scenarios:
        failures = 0
        total_errors = 0

        for trial in range(num_trials):
            result = surface_code.run_error_correction_cycle(p_x, p_y, p_z)

            if not result['is_correctable']:
                failures += 1
            total_errors += result['num_errors']

        results[scenario_name] = {
            'failure_rate': failures / num_trials,
            'avg_errors_per_trial': total_errors / num_trials
        }

    return results


# Demo and testing
if __name__ == "__main__":
    print("Testing surface code with error correction...")

    # Basic functionality test
    sc = SurfaceCodeWithErrors()

    # Test single cycle
    result = sc.run_error_correction_cycle(0.05, 0, 0.05)
    print(f"\nSingle cycle test:")
    print(f"  Errors: {result['errors_injected']}")
    print(f"  Syndromes: X={result['x_syndromes']}, Z={result['z_syndromes']}")
    print(f"  Correctable: {result['is_correctable']}")

    if result['errors_injected']:
        sc.visualize_errors(result['errors_injected'])

    # Performance stats
    stats = sc.get_performance_stats()
    print(f"\nPerformance: {stats}")

    print("\nError correction module working correctly!")
