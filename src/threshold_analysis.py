"""
Threshold analysis for distributed quantum error correction.

Compares error correction thresholds between centralized and distributed 
surface code architectures using Monte Carlo simulation.
"""

import numpy as np
from typing import List, Dict, Tuple
from error_correction import SurfaceCodeWithErrors
from distributed_architecture import DistributedSurfaceCode

class ThresholdAnalyzer:
    """Monte Carlo threshold analysis for surface codes."""

    def __init__(self):
        self.centralized_code = SurfaceCodeWithErrors()
        self.distributed_code = DistributedSurfaceCode()

    def run_threshold_scan(self, error_rates: List[float], trials_per_rate: int = 30) -> Dict:
        """
        Run threshold analysis across multiple error rates.

        Args:
            error_rates: List of physical error rates to test
            trials_per_rate: Number of Monte Carlo trials per rate

        Returns:
            Dictionary with logical error rates for each architecture
        """

        results = {
            'error_rates': error_rates,
            'centralized_logical_rates': [],
            'distributed_logical_rates': [],
            'centralized_std_errors': [],
            'distributed_std_errors': []
        }

        for rate in error_rates:
            # Test centralized architecture
            cent_failures = 0
            for _ in range(trials_per_rate):
                result = self.centralized_code.run_error_correction_cycle(rate, 0, rate)
                if not result['is_correctable'] or result['total_syndrome_weight'] > 3:
                    cent_failures += 1

            cent_logical_rate = cent_failures / trials_per_rate
            cent_std_error = np.sqrt(cent_logical_rate * (1 - cent_logical_rate) / trials_per_rate)

            results['centralized_logical_rates'].append(cent_logical_rate)
            results['centralized_std_errors'].append(cent_std_error)

            # Test distributed architecture
            dist_failures = 0
            for _ in range(trials_per_rate):
                result = self.distributed_code.run_distributed_cycle(
                    intra_rate=rate * 0.3,
                    inter_rate=rate * 2.0,
                    network_effects=True
                )
                if result['syndrome_weight'] > 3 or result['total_errors'] > 3:
                    dist_failures += 1

            dist_logical_rate = dist_failures / trials_per_rate
            dist_std_error = np.sqrt(dist_logical_rate * (1 - dist_logical_rate) / trials_per_rate)

            results['distributed_logical_rates'].append(dist_logical_rate)
            results['distributed_std_errors'].append(dist_std_error)

        return results

    def estimate_thresholds(self, results: Dict, target_logical_rate: float = 0.1) -> Dict[str, float]:
        """
        Estimate error correction thresholds from scan results.

        Args:
            results: Results from run_threshold_scan()
            target_logical_rate: Target logical error rate for threshold

        Returns:
            Dictionary with estimated thresholds
        """
        thresholds = {}

        error_rates = np.array(results['error_rates'])

        # Centralized threshold
        cent_rates = np.array(results['centralized_logical_rates'])
        valid_cent = error_rates[cent_rates < target_logical_rate]
        if len(valid_cent) > 0:
            thresholds['centralized'] = valid_cent[-1]  # Highest valid rate
        else:
            # Linear extrapolation
            if len(cent_rates) >= 2:
                slope = (cent_rates[-1] - cent_rates[0]) / (error_rates[-1] - error_rates[0])
                intercept = cent_rates[0] - slope * error_rates[0]
                thresholds['centralized'] = (target_logical_rate - intercept) / slope
                thresholds['centralized'] = max(0.001, thresholds['centralized'])  # Floor

        # Distributed threshold
        dist_rates = np.array(results['distributed_logical_rates'])
        valid_dist = error_rates[dist_rates < target_logical_rate]
        if len(valid_dist) > 0:
            thresholds['distributed'] = valid_dist[-1]
        else:
            # Linear extrapolation
            if len(dist_rates) >= 2:
                slope = (dist_rates[-1] - dist_rates[0]) / (error_rates[-1] - error_rates[0])
                intercept = dist_rates[0] - slope * error_rates[0]
                thresholds['distributed'] = (target_logical_rate - intercept) / slope
                thresholds['distributed'] = max(0.001, thresholds['distributed'])  # Floor

        return thresholds

    def calculate_degradation(self, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Calculate threshold degradation metrics."""
        if 'centralized' not in thresholds or 'distributed' not in thresholds:
            return {}

        cent_thresh = thresholds['centralized']
        dist_thresh = thresholds['distributed']

        relative_degradation = (cent_thresh - dist_thresh) / cent_thresh
        absolute_degradation = cent_thresh - dist_thresh

        return {
            'relative_degradation': relative_degradation,
            'absolute_degradation': absolute_degradation,
            'distributed_efficiency': dist_thresh / cent_thresh
        }

class ThresholdVisualizer:
    """Visualization tools for threshold analysis."""

    @staticmethod
    def plot_threshold_comparison(results: Dict, thresholds: Dict = None):
        """Plot threshold comparison between architectures."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        error_rates = results['error_rates']
        cent_rates = results['centralized_logical_rates']
        dist_rates = results['distributed_logical_rates']
        cent_errors = results.get('centralized_std_errors', [0] * len(error_rates))
        dist_errors = results.get('distributed_std_errors', [0] * len(error_rates))

        plt.figure(figsize=(10, 6))

        # Main threshold plot
        plt.subplot(1, 2, 1)
        plt.errorbar(error_rates, cent_rates, yerr=cent_errors, 
                    label='Centralized', marker='o', capsize=3)
        plt.errorbar(error_rates, dist_rates, yerr=dist_errors,
                    label='Distributed', marker='s', capsize=3)

        plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='Threshold')
        plt.xlabel('Physical Error Rate')
        plt.ylabel('Logical Error Rate')
        plt.title('Surface Code Thresholds')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # Performance ratio plot
        plt.subplot(1, 2, 2)
        ratios = [d/max(c, 1e-10) for c, d in zip(cent_rates, dist_rates)]
        plt.plot(error_rates, ratios, 'ro-', linewidth=2)
        plt.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
        plt.xlabel('Physical Error Rate')
        plt.ylabel('Performance Ratio (Dist/Cent)')
        plt.title('Distributed Overhead')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def run_comprehensive_analysis():
    """Run comprehensive threshold analysis."""
    print("Comprehensive Threshold Analysis")
    print("=" * 35)

    analyzer = ThresholdAnalyzer()

    # Define scan parameters
    error_rates = [0.005, 0.010, 0.015, 0.020, 0.025]
    trials = 25

    print(f"Scanning {len(error_rates)} error rates with {trials} trials each")

    # Run threshold scan
    results = analyzer.run_threshold_scan(error_rates, trials)

    # Estimate thresholds
    thresholds = analyzer.estimate_thresholds(results)
    degradation = analyzer.calculate_degradation(thresholds)

    # Report results
    print("\nThreshold Results:")
    for arch, threshold in thresholds.items():
        print(f"  {arch.capitalize()}: {threshold:.4f}")

    if degradation:
        print(f"\nPerformance Impact:")
        print(f"  Threshold degradation: {degradation['relative_degradation']:.1%}")
        print(f"  Distributed efficiency: {degradation['distributed_efficiency']:.1%}")

    # Performance summary
    cent_avg = np.mean(results['centralized_logical_rates'])
    dist_avg = np.mean(results['distributed_logical_rates'])

    print(f"\nAverage Performance:")
    print(f"  Centralized logical error rate: {cent_avg:.3f}")
    print(f"  Distributed logical error rate: {dist_avg:.3f}")
    print(f"  Average penalty: {dist_avg/cent_avg:.2f}x")

    # Generate plots
    visualizer = ThresholdVisualizer()
    visualizer.plot_threshold_comparison(results, thresholds)

    return results, thresholds, degradation

if __name__ == "__main__":
    run_comprehensive_analysis()
