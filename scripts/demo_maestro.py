"""Demonstration of MaestroBackend for simulation and exact reference evaluation.

Shows:
1. Exact check-subspace reduction for 37-qubit colour code d=7 in ~0.1 s.
2. Dynamic circuit simulation with mid-circuit measurements and feed-forward.
3. Native exact observable expectation value estimation without shot noise.
"""

from __future__ import annotations

import networkx as nx
from qecbench import Chain, codes
from qecbench.backends import MaestroBackend


def main():
    print("=" * 70)
    print("MaestroBackend Demonstration")
    print("=" * 70)

    # 1. Initialize the backend
    backend = MaestroBackend(graph=nx.path_graph(9), seed=42)
    print(f"Backend name:    {backend.name}")
    print(f"Vendor:          {backend.vendor}")
    print(f"Simulated:       {backend.simulated}")

    # 2. Exact Check-Subspace Reduction (Fast Reference Evaluation)
    print("\n--- 1. Exact Reference Energy (Check-Subspace Reduction) ---")
    code7 = codes.color_code(7)
    print(f"Code: {code7.name} ({code7.n_data} physical data qubits, {code7.n_checks} checks)")

    # 37 qubits at depth 10 computes analytically in ~0.1s without truncation:
    energy_d7_p10 = backend.ideal_energy(code7, depth=10)
    print(f"Depth 10 Exact Energy: {energy_d7_p10:.6f}")

    # Also compute depth 1, 3, and 5:
    for p in (1, 3, 5):
        print(f"Depth {p:2d} Exact Energy: {backend.ideal_energy(code7, depth=p):.6f}")

    # 3. Dynamic Circuit Execution (Mid-Circuit Measurements + Feed-Forward)
    print("\n--- 2. Native Dynamic Circuit Execution (MCM + Feed-Forward) ---")
    chain = Chain((0, 1, 2))
    qc = backend.build([chain], depth=2, delta=0.5, kind="mcm")
    print(f"Built MCM circuit with {qc.num_qubits} qubits and {qc.num_clbits} clbits")

    result = backend.submit([qc], shots=500)
    print(f"Job ID: {result['job_id']}")
    print(f"Measured outcome counts (500 shots): {result['counts'][0]}")

    # 4. Exact Observable Estimation (Zero Shot Noise)
    print("\n--- 3. Exact Observable Estimation ---")
    code3 = codes.color_code(3)
    from qecbench.lrqaoa import ideal_circuit
    qc_d3 = ideal_circuit(code3.hamiltonian, code3.n_data, depth=1)
    observables = [
        "".join("Z" if q in term else "I" for q in range(code3.n_data))
        for term in code3.checks
    ]
    exp_vals = backend.estimate_observables(qc_d3, observables)
    print(f"Colour code d=3 checks expectation values: {[round(v, 4) for v in exp_vals]}")

    # 5. Nearest-Neighbour Ladder Circuit & GPU Mode
    print("\n--- 4. Nearest-Neighbour Ladder & GPU Configuration ---")
    from qecbench.backends.maestro import build_ladder_circuit
    ladder_qc = build_ladder_circuit(code3, depth=1)
    print(f"Nearest-neighbour ladder gates for {code3.name}: {dict(ladder_qc.count_ops())}")
    energy_ladder = backend.ideal_energy(code3, depth=1, use_subspace=False, ladder=True)
    print(f"Colour code d=3 ladder simulation energy: {energy_ladder:.6f}")
    print("GPU acceleration: supported via MaestroBackend(gpu=True) with CUDA/A100")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
