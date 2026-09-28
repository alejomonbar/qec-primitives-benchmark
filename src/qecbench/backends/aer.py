"""Local simulation: the whole plan/submit/harvest loop for free.

* ``AerBackend`` - an arbitrary coupling graph, circuits on compact virtual qubits.
  ``dialect="iqm"`` builds the IQM native circuit and runs it through ``iqm_to_dynamic``.
* ``simulate_job`` - what ``IBMBackend(local=True)`` and ``IQMBackend(local=True)`` call, so a
  rehearsal exercises the real vendor build path (for IBM: pinned transpilation onto a fake
  device, simulated with that device's noise on the physical qubits).
"""

from __future__ import annotations

import networkx as nx

from ..circuits import build_dynamic, build_iqm, iqm_to_dynamic, split_register_counts
from ..layout import synthetic_graph, validate_batch
from .base import Backend, reverse_keys


def run_aer(circuits, shots, noise_model=None, seed=None):
    """Run circuits on Aer; returns one ``{register: counts}`` per circuit.

    Aer truncates idle qubits, so the method is chosen on the qubits actually used.  Each
    circuit gets its own seed, derived from ``(seed, i)``: identical circuits must not share a
    random stream, and consecutive seeds would, since Aer seeds shot ``k`` with ``seed + k``.
    """
    import numpy as np
    from qiskit_aer import AerSimulator

    out = []
    for i, qc in enumerate(circuits):
        active = len({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
        sim = AerSimulator(method="matrix_product_state" if active > 20 else "automatic",
                           noise_model=noise_model)
        circuit_seed = None if seed is None else int(np.random.SeedSequence([seed, i]).generate_state(1)[0] >> 1)
        run = sim.run(qc, shots=shots, seed_simulator=circuit_seed)
        out.append(split_register_counts(run.result().get_counts(), qc))
    return out


def local_job_id(name):
    """A job id unique to this simulation run, so repeat runs are harvested, not skipped."""
    from datetime import datetime

    return f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"


def simulate_job(circuits, shots, noise_model=None, seed=None, job_id="local"):
    """A manifest record whose counts are already inside it (``fetch`` reads them back)."""
    counts = []
    for qc, regs in zip(circuits, run_aer(circuits, shots, noise_model, seed)):
        n_inst = sum(1 for c in qc.cregs if c.name.startswith("d"))
        counts.append([reverse_keys(regs[f"d{k}"]) for k in range(n_inst)])
    return {"job_id": job_id, "counts": counts}


def stored_counts(record, tasks):
    return [record["counts"][t["index"]] for t in tasks]


class AerBackend(Backend):
    vendor = "aer"
    is_simulator = True
    max_circuits_per_job = 10 ** 9

    def __init__(self, graph=None, name="aer_simulator", dialect="dynamic", noise_model=None,
                 seed=None, rules=None):
        super().__init__(name)
        self.graph = graph if graph is not None else nx.path_graph(9)
        self.dialect = dialect
        self.noise_model = noise_model
        self.seed = seed
        self.rules = rules            # optional adapter whose validate() should also apply
        self.kinds = ("mcm", "direct")
        self.frames = ("Z", "X") if dialect == "dynamic" else ("Z",)
        self._submitted = 0

    def coupling_graph(self):
        return self.graph

    def feedforward_groups(self):
        return self.rules.feedforward_groups() if self.rules else None

    def validate(self, batch, kind="mcm"):
        if self.rules is not None:
            return self.rules.validate(batch, kind)
        return validate_batch(batch, self.graph)

    def build(self, batch, depth, delta, kind="mcm", frame="Z"):
        if self.dialect == "iqm":
            if frame != "Z":
                raise ValueError("the IQM dialect is built in the Z frame only")
            return iqm_to_dynamic(build_iqm(batch, depth, delta, kind))
        return build_dynamic(batch, depth, delta, kind, frame=frame)

    @property
    def noise_description(self):
        return None if self.noise_model is None else "qiskit_aer noise model supplied by the caller"

    def submit(self, circuits, shots):
        seed = None if self.seed is None else self.seed + self._submitted
        self._submitted += len(circuits)
        return simulate_job(circuits, shots, self.noise_model, seed, job_id=local_job_id(self.name))

    def fetch(self, record, tasks):
        return stored_counts(record, tasks)


class SimBackend(AerBackend):
    """A whole benchmark with no vendor account: synthetic chip, uniform depolarizing noise.

    The chip is a ``heavy_hex`` lattice of ``qubits`` qubits (see ``layout.synthetic_graph``)
    and the noise is the same everywhere (see ``qecbench.noise``), so this answers "what does
    the benchmark look like at this error rate?" rather than "what will device X return?" -
    for the latter use ``IBMBackend(..., local=True)``, which takes the real calibration.

    The noise parameters are arguments rather than notebook knobs::

        SimBackend()                       # the defaults of qecbench.noise
        SimBackend(p_2q=0.01)              # a worse two-qubit gate
        SimBackend(p_1q=0, p_2q=0, readout=0)   # noiseless, to check the plumbing
    """

    vendor = "sim"

    def __init__(self, name="noisy_simulator", qubits=54, topology="heavy_hex", seed=None,
                 dialect="dynamic", rules=None, **noise):
        from ..noise import DEFAULTS, depolarizing_noise_model, describe

        unknown = set(noise) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown noise parameter(s) {sorted(unknown)}; "
                            f"expected any of {sorted(DEFAULTS)}")
        self.noise_params = {**DEFAULTS, **noise}
        super().__init__(graph=synthetic_graph(topology, qubits), name=name, dialect=dialect,
                         noise_model=depolarizing_noise_model(**self.noise_params), seed=seed,
                         rules=rules)
        self._description = describe(**self.noise_params)

    @property
    def noise_description(self):
        return self._description
