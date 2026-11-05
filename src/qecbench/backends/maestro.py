"""Local, noiseless simulation on Maestro, through ``qiskit-maestro``.

* ``MaestroBackend`` - ``AerBackend`` with ``qiskit_maestro.MaestroBackend`` in place of
  ``AerSimulator``: the same coupling graph, dialects and plan/submit/harvest loop.
* ``maestro_energy`` - the exact LR-QAOA energy from ``qiskit_maestro.MaestroEstimator``.

Simulator options use Aer's names (``method``, ``device``, ``max_bond_dimension``, ...).
Aer noise models are not supported.
"""

from __future__ import annotations

from ..circuits import split_register_counts
from .aer import AerBackend, local_job_id
from .base import reverse_keys


def _method(qc, method):
    """Statevector up to 20 active qubits, MPS above, unless ``method`` is given."""
    if method is not None:
        return method
    active = len({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    return "matrix_product_state" if active > 20 else "statevector"


def run_maestro(circuits, shots, method=None, seed=None, **backend_options):
    """Run circuits on Maestro; returns one ``{register: counts}`` per circuit.

    Seeds are derived per circuit from ``(seed, i)``, as in ``run_aer``.
    """
    import numpy as np
    from qiskit_maestro import MaestroBackend as MaestroSimulator

    out = []
    for i, qc in enumerate(circuits):
        sim = MaestroSimulator(method=_method(qc, method), **backend_options)
        circuit_seed = None if seed is None else int(np.random.SeedSequence([seed, i]).generate_state(1)[0] >> 1)
        run = sim.run(qc, shots=shots, seed_simulator=circuit_seed)
        out.append(split_register_counts(run.result().get_counts(), qc))
    return out


def estimate_maestro(circuit, observables, method=None, **backend_options):
    """Exact expectation values (no shots) of ``observables``, e.g. a ``SparsePauliOp``."""
    from qiskit_maestro import MaestroEstimator

    estimator = MaestroEstimator(method=_method(circuit, method), **backend_options)
    return estimator.run([(circuit, observables)]).result()[0].data.evs


def maestro_energy(code, depth, delta=0.5, method=None, **backend_options) -> float:
    """Ideal LR-QAOA energy ``sum_c w_c <Z_c>``, exact via ``MaestroEstimator``."""
    from qiskit.quantum_info import SparsePauliOp

    from ..lrqaoa import ideal_circuit

    qc = ideal_circuit(code.hamiltonian, code.n_data, depth, delta)
    hamiltonian = SparsePauliOp.from_sparse_list(
        [("Z" * len(term), term, w) for term, w in code.hamiltonian.items()], code.n_data)
    return float(estimate_maestro(qc, hamiltonian, method, **backend_options))


class MaestroBackend(AerBackend):
    """``AerBackend`` on Maestro.

    ``method`` defaults per circuit to statevector up to 20 active qubits and MPS above;
    ``device="GPU"`` needs Maestro's GPU plugin.  Other keywords (``max_bond_dimension``,
    ``truncation_threshold``, ``maestro_options``, ...) go to ``qiskit_maestro.MaestroBackend``.
    """

    vendor = "maestro"

    def __init__(self, graph=None, name="maestro_simulator", dialect="dynamic", method=None,
                 device="CPU", seed=None, rules=None, **backend_options):
        super().__init__(graph=graph, name=name, dialect=dialect, seed=seed, rules=rules)
        self.method = method
        self.backend_options = {"device": device, **backend_options}

    def submit(self, circuits, shots):
        seed = None if self.seed is None else self.seed + self._submitted
        self._submitted += len(circuits)
        counts = []
        for qc, regs in zip(circuits, run_maestro(circuits, shots, self.method, seed,
                                                  **self.backend_options)):
            n_inst = sum(1 for c in qc.cregs if c.name.startswith("d"))
            counts.append([reverse_keys(regs[f"d{k}"]) for k in range(n_inst)])
        return {"job_id": local_job_id(self.name), "counts": counts}

    def ideal_energy(self, code, depth, delta=0.5) -> float:
        return maestro_energy(code, depth, delta, self.method, **self.backend_options)
