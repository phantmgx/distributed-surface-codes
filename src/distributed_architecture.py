"""
Distributed surface code architecture for quantum computing networks.

Models performance trade-offs between centralized and distributed quantum 
computer architectures using network overhead and inter-module communication.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from error_correction import SurfaceCodeWithErrors

class DistributedSurfaceCode(SurfaceCodeWithErrors):
    """
    Surface code adapted for distributed quantum computing architecture.

    Models a 2-module distributed system with network communication overhead.
    """

    def __init__(self):
        super().__init__()

        # System partitioning
        self.modules = {
            'module_1': [0, 1, 3, 4],      # Top-left block
            'module_2': [2, 5, 6, 7, 8]    # Remaining qubits
        }

        # Inter-module communication links
        self.inter_module_links = [
            (1, 2),   # Module 1 to Module 2 connections
            (4, 5),   
            (3, 6),   
            (4, 7)    
        ]

        # Error rate parameters
        self.intra_module_rate = 0.001  # Low rate within modules
        self.inter_module_rate = 0.01   # Higher rate between modules

        # Network overhead modeling (18% penalty per link)
        self.network_overhead = 1.0 + 0.18 * len(self.inter_module_links)

    def inject_distributed_errors(self, intra_rate: float, inter_rate: float) -> List[Tuple[int, str]]:
        """
        Inject errors with different rates for intra vs inter-module qubits.
        """
        errors_applied = []

        # Intra-module errors (lower rate)
        for module_qubits in self.modules.values():
            for qubit in module_qubits:
                if qubit in self.data_qubits and random.random() < intra_rate:
                    error_type = random.choice(['X', 'Z'])

                    if error_type == 'X':
                        self.circuit.x(self.qreg[qubit])
                    else:
                        self.circuit.z(self.qreg[qubit])

                    errors_applied.append((qubit, error_type))

        # Inter-module link errors (higher rate with network overhead)
        effective_inter_rate = inter_rate * self.network_overhead

        for q1, q2 in self.inter_module_links:
            if random.random() < effective_inter_rate:
                # Apply error to one of the linked qubits
                target_qubit = random.choice([q1, q2])

                if target_qubit in self.data_qubits:
                    error_type = random.choice(['X', 'Z'])

                    if error_type == 'X':
                        self.circuit.x(self.qreg[target_qubit])
                    else:
                        self.circuit.z(self.qreg[target_qubit])

                    errors_applied.append((target_qubit, error_type))

        if errors_applied:
            self.circuit.barrier(label="distributed_errors")

        return errors_applied

    def run_distributed_cycle(self, intra_rate: float = 0.001, inter_rate: float = 0.01, 
                            network_effects: bool = True) -> Dict:
        """
        Run complete distributed error correction cycle.

        Args:
            intra_rate: Error rate within modules
            inter_rate: Base error rate between modules
            network_effects: Whether to apply network overhead

        Returns:
            Dictionary with cycle results and performance metrics
        """
        # Reset and prepare
        self.reset_circuit()
        self.encode_logical_zero()

        # Apply network effects
        if network_effects:
            effective_inter_rate = inter_rate * self.network_overhead
        else:
            effective_inter_rate = inter_rate

        # Inject distributed errors
        errors = self.inject_distributed_errors(intra_rate, effective_inter_rate)

        # Categorize errors by location
        module_breakdown = {'module_1': [], 'module_2': [], 'inter_module': []}

        for qubit, error_type in errors:
            if qubit in self.modules['module_1']:
                module_breakdown['module_1'].append((qubit, error_type))
            elif qubit in self.modules['module_2']:
                module_breakdown['module_2'].append((qubit, error_type))
            else:
                # Check if it's on an inter-module link
                for q1, q2 in self.inter_module_links:
                    if qubit in [q1, q2]:
                        module_breakdown['inter_module'].append((qubit, error_type))
                        break

        # Calculate syndromes
        x_syndromes, z_syndromes = self.calculate_syndromes(errors)
        syndrome_weight = sum(x_syndromes) + sum(z_syndromes)

        return {
            'errors_by_location': module_breakdown,
            'total_errors': len(errors),
            'intra_module_errors': len(module_breakdown['module_1']) + len(module_breakdown['module_2']),
            'inter_module_errors': len(module_breakdown['inter_module']),
            'x_syndromes': x_syndromes,
            'z_syndromes': z_syndromes,
            'syndrome_weight': syndrome_weight,
            'network_overhead': self.network_overhead if network_effects else 1.0
        }

class NetworkAnalyzer:
    """Analysis tools for distributed vs centralized performance."""

    def __init__(self):
        self.centralized_code = SurfaceCodeWithErrors()
        self.distributed_code = DistributedSurfaceCode()

    def compare_architectures(self, num_trials: int = 20, base_error_rate: float = 0.015) -> Dict:
        """Compare centralized vs distributed performance."""

        centralized_results = []
        distributed_results = []

        for _ in range(num_trials):
            # Centralized test
            cent_result = self.centralized_code.run_error_correction_cycle(
                base_error_rate, 0, base_error_rate
            )
            centralized_results.append({
                'errors': cent_result['num_errors'],
                'syndrome_weight': cent_result['total_syndrome_weight'],
                'correctable': cent_result['is_correctable']
            })

            # Distributed test (with realistic rate scaling)
            dist_result = self.distributed_code.run_distributed_cycle(
                intra_rate=base_error_rate * 0.3,
                inter_rate=base_error_rate * 2.0,
                network_effects=True
            )
            distributed_results.append({
                'errors': dist_result['total_errors'],
                'syndrome_weight': dist_result['syndrome_weight'],
                'network_overhead': dist_result['network_overhead']
            })

        # Calculate averages
        cent_avg_errors = np.mean([r['errors'] for r in centralized_results])
        cent_avg_syndromes = np.mean([r['syndrome_weight'] for r in centralized_results])
        cent_success_rate = np.mean([r['correctable'] for r in centralized_results])

        dist_avg_errors = np.mean([r['errors'] for r in distributed_results])
        dist_avg_syndromes = np.mean([r['syndrome_weight'] for r in distributed_results])
        dist_avg_overhead = np.mean([r['network_overhead'] for r in distributed_results])

        # Calculate degradation
        error_degradation = (dist_avg_errors - cent_avg_errors) / max(cent_avg_errors, 0.1)
        syndrome_degradation = (dist_avg_syndromes - cent_avg_syndromes) / max(cent_avg_syndromes, 0.1)

        return {
            'centralized': {
                'avg_errors': cent_avg_errors,
                'avg_syndromes': cent_avg_syndromes,
                'success_rate': cent_success_rate
            },
            'distributed': {
                'avg_errors': dist_avg_errors,
                'avg_syndromes': dist_avg_syndromes,
                'avg_overhead': dist_avg_overhead
            },
            'performance_impact': {
                'error_increase': error_degradation,
                'syndrome_increase': syndrome_degradation,
                'network_penalty': dist_avg_overhead
            }
        }

def demo_distributed_system():
    """Demonstration of distributed surface code system."""
    print("Distributed Surface Code Demo")
    print("-" * 30)

    # Setup
    dist_sc = DistributedSurfaceCode()
    analyzer = NetworkAnalyzer()

    print(f"System configuration:")
    print(f"  Modules: {len(dist_sc.modules)}")
    print(f"  Inter-module links: {len(dist_sc.inter_module_links)}")
    print(f"  Network overhead: {dist_sc.network_overhead:.2f}x")

    # Single cycle test
    print("\nTesting distributed cycle:")
    result = dist_sc.run_distributed_cycle(0.002, 0.012)

    print(f"  Total errors: {result['total_errors']}")
    print(f"  Syndrome weight: {result['syndrome_weight']}")
    print(f"  Network overhead: {result['network_overhead']:.2f}x")

    # Architecture comparison
    print("\nArchitecture comparison:")
    comparison = analyzer.compare_architectures(num_trials=10)

    print(f"  Centralized avg errors: {comparison['centralized']['avg_errors']:.1f}")
    print(f"  Distributed avg errors: {comparison['distributed']['avg_errors']:.1f}")
    print(f"  Performance penalty: {comparison['performance_impact']['network_penalty']:.2f}x")

    return dist_sc, analyzer

if __name__ == "__main__":
    demo_distributed_system()
