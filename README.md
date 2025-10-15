# Distributed Quantum Error Correction: A Threshold Analysis

A comprehensive study investigating performance trade-offs in distributed quantum computing architectures using surface codes, combining classical and quantum simulations to quantify network effects on error correction thresholds.

## Key Research Findings

Breakthrough Results:
- 60% threshold degradation: Centralized (2.5%) vs Distributed (1.0%) 
- 4.5× performance penalty: Average logical error rate increase in distributed systems
- 1.72× network overhead: Inter-module communication bottleneck quantified
- Network communication identified as primary limiting factor for distributed quantum systems

Technical Contributions:
- Systematic comparison of centralized vs distributed quantum error correction
- Quantum simulation validation using Qiskit AerSimulator  
- Statistical analysis with Monte Carlo methodology
- Network effects quantification in modular quantum architectures

## Installation & Setup

Prerequisites:
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

Quick Setup:
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Start Jupyter: jupyter notebook distributed_quantum_error_correction.ipynb

Alternative Setup:
- Install requirements manually: pip install numpy matplotlib qiskit qiskit-aer jupyter networkx scipy
- Run the notebook: jupyter notebook distributed_quantum_error_correction.ipynb

## Research Overview

This study addresses fundamental questions in distributed quantum computing:

1. How do inter-module communication delays affect quantum error correction?
2. What is the threshold degradation in distributed vs centralized systems?  
3. Can quantum simulation reveal insights missed by classical analysis?

Methodology:
- 3×3 surface code implementation with stabilizer measurements
- Classical syndrome analysis and MWMP decoding
- Quantum circuit simulation using Qiskit AerSimulator
- Distributed architecture modeling with network overhead
- Monte Carlo threshold analysis with statistical validation

## Key Results

Quantified Performance Impact:
- Threshold degradation: 60% reduction in error correction capability
- Performance penalty: 4.5× increase in logical error rates
- Network bottleneck: 1.72× communication overhead factor
- Critical finding: Inter-module error rates must be <2× intra-module rates for viability

Research Implications:
- Network communication significantly impacts error correction capability
- Distributed systems require higher component fidelity to match centralized performance  
- Network optimization critical for modular quantum system design
- Trade-off quantified between system modularity and error correction performance

## Dependencies

All required modules are included in the src/ directory. The notebook automatically imports them using relative paths.

Core Requirements:
- numpy>=1.21.0 - Numerical computations
- matplotlib>=3.5.0 - Visualization  
- qiskit>=0.45.0 - Quantum simulation
- qiskit-aer>=0.12.0 - Quantum circuit simulation
- jupyter>=1.0.0 - Notebook environment

## Usage

1. Install dependencies: pip install -r requirements.txt
2. Launch Jupyter: jupyter notebook distributed_quantum_error_correction.ipynb
3. Run all cells: Execute from top to bottom for complete analysis
4. View results: Observe threshold analysis plots and quantified metrics



This work establishes quantitative foundations for distributed quantum computing architecture design.

