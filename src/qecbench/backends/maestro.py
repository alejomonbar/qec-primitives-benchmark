"""Local, noiseless simulation with qoro-maestro 0.3.5.

Qiskit builds the benchmark circuits; OpenQASM 3 preserves mid-circuit measurements
and classical feed-forward when translating them to Maestro's native circuits.
No Aer simulator is used. Aer noise models are not supported by this adapter.
"""

from __future__ import annotations

import copy
from numbers import Integral

import networkx as nx
import numpy as np

from ..circuits import build_dynamic, build_iqm, iqm_to_dynamic
from .aer import local_job_id, stored_counts
from .base import Backend, reverse_keys


def check_rank(checks) -> int:
    """Return the GF(2) rank of the binary check matrix using bitwise elimination."""
    basis = {}
    for term in checks:
        value = sum(1 << int(q) for q in term)
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot]
            else:
                basis[pivot] = value
                break
    return len(basis)


def evolve_check_subspace(code, depth: int, delta: float = 0.5) -> np.ndarray:
    """Return amplitudes in |b> = prod_j P_j**b_j |+>**n (b_j is bit j).

    Independent binary check rows make these states orthonormal. On this basis,
    P_j acts as X_j and physical X_q acts as prod_{j:q in check_j} Z_j.
    Both alternating layers preserve the subspace, whose initial state is |0>.
    """
    from ..lrqaoa import angles

    rank = check_rank(code.checks)
    if rank != code.n_checks:
        raise ValueError("check subspace requires linearly independent Z checks")
    if rank > 22:
        raise ValueError("check subspace limits the effective statevector to 22 qubits")
    gammas, betas = angles(depth, delta)
    labels = np.arange(1 << rank, dtype=np.uint32)
    mixer = np.zeros(1 << rank, dtype=float)
    for q in range(code.n_data):
        parity = np.zeros(1 << rank, dtype=np.uint32)
        for j, term in enumerate(code.checks):
            if q in term:
                parity ^= (labels >> j) & 1
        mixer += 1 - 2 * parity.astype(float)
    state = np.zeros(1 << rank, dtype=np.complex128)
    state[0] = 1.0
    for gamma, beta in zip(gammas, betas):
        for j, weight in enumerate(code.weights):
            blocks = state.reshape(-1, 2, 1 << j)
            lo, hi = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
            co, si = np.cos(gamma * weight), -1j * np.sin(gamma * weight)
            blocks[:, 0, :] = co * lo + si * hi
            blocks[:, 1, :] = si * lo + co * hi
        state *= np.exp(1j * beta * mixer)
    return state


def check_subspace_energy(code, depth: int, delta: float = 0.5) -> float:
    """Exact energy in the check-generated subspace without state truncation or shot noise."""
    state = evolve_check_subspace(code, depth, delta)
    energy = 0.0
    for j, weight in enumerate(code.weights):
        blocks = state.reshape(-1, 2, 1 << j)
        energy += weight * 2 * np.vdot(blocks[:, 0, :], blocks[:, 1, :]).real
    return float(energy)


def estimate_maestro(circuit, observables: list[str], config=None):
    """Evaluate exact expectation values without shots using Maestro's native estimate API."""
    try:
        import maestro
    except ImportError as exc:
        raise ImportError('Maestro simulation requires pip install -e ".[maestro]"') from exc
    from qiskit import qasm3

    if config is not None and not isinstance(config, maestro.SimulatorConfig):
        raise TypeError("config must be a maestro.SimulatorConfig")
    cfg = copy.copy(config) if config is not None else maestro.SimulatorConfig()
    if config is None and circuit.num_qubits > 20:
        cfg.simulation_type = maestro.SimulationType.MatrixProductState

    if getattr(cfg, "simulator_type", None) == getattr(maestro.SimulatorType, "Gpu", None):
        maestro.init_gpu()

    mc = maestro.QasmToCirc().parse_and_translate(qasm3.dumps(circuit))
    result = mc.estimate(observables, config=cfg)
    return result["expectation_values"]


def build_ladder_circuit(code, depth: int, delta: float = 0.5, order: list[int] | None = None):
    """Build a 1D nearest-neighbour CNOT ladder circuit along an MPS site ordering.

    Decomposes each multi-qubit check into a local CNOT cascade, minimizing MPS bond entanglement.
    """
    from qiskit import QuantumCircuit
    from ..lrqaoa import angles

    n = code.n_data
    qc = QuantumCircuit(n)
    qc.h(range(n))
    pos = {q: i for i, q in enumerate(order)} if order is not None else {q: q for q in range(n)}
    for gamma, beta in zip(*angles(depth, delta)):
        for term, weight in code.hamiltonian.items():
            support = sorted(term, key=lambda q: pos[q])
            pairs = list(zip(support[:-1], support[1:]))
            for u, v in pairs:
                qc.cx(u, v)
            qc.rz(2 * weight * gamma, support[-1])
            for u, v in reversed(pairs):
                qc.cx(u, v)
        qc.rx(-2 * beta, range(n))
    return qc


def maestro_energy(code, depth: int, delta: float = 0.5, config=None, use_subspace: bool = True,
                   ladder: bool = False, order: list[int] | None = None) -> float:
    """Return ideal LR-QAOA energy, using the exact check-subspace reduction when eligible."""
    if use_subspace and not ladder and order is None and delta == 0.5 and getattr(code, "checks", None) is not None:
        try:
            return check_subspace_energy(code, depth, delta)
        except (ValueError, AttributeError):
            pass

    if ladder or order is not None:
        qc = build_ladder_circuit(code, depth, delta, order)
    else:
        from ..lrqaoa import ideal_circuit
        qc = ideal_circuit(code.hamiltonian, code.n_data, depth, delta)

    observables = [
        "".join("Z" if q in term else "I" for q in range(code.n_data))
        for term in code.checks
    ]
    exp_vals = estimate_maestro(qc, observables, config=config)
    return float(np.dot(code.weights, exp_vals))


def run_maestro(circuits, shots, config=None, seed=None):
    """Return ``{register: counts}`` per circuit, in Qiskit's register bit order.

    ``config`` is a native ``maestro.SimulatorConfig`` and is never modified.
    Without it, use QCSim statevector up to 20 active qubits and MPS above that.
    An explicit seed overrides the config seed; circuits get distinct child seeds.
    """
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    try:
        import maestro
    except ImportError as exc:
        raise ImportError('Maestro simulation requires pip install -e ".[maestro]"') from exc
    from qiskit import qasm3

    if config is not None and not isinstance(config, maestro.SimulatorConfig):
        raise TypeError("config must be a maestro.SimulatorConfig")
    if getattr(config, "simulator_type", None) == getattr(maestro.SimulatorType, "Gpu", None):
        maestro.init_gpu()
    base_seed = seed if seed is not None else getattr(config, "seed", None)
    out = []
    for i, qc in enumerate(circuits):
        cfg = copy.copy(config) if config is not None else maestro.SimulatorConfig()
        if config is None:
            active = len({q for inst in qc.data for q in inst.qubits})
            if active > 20:
                cfg.simulation_type = maestro.SimulationType.MatrixProductState
        if base_seed is not None:
            cfg.seed = int(np.random.SeedSequence([base_seed, i]).generate_state(1)[0] >> 1)
        circuit = maestro.QasmToCirc().parse_and_translate(qasm3.dumps(qc))
        counts = circuit.execute(config=cfg, shots=int(shots))["counts"]
        registers = {reg.name: {} for reg in qc.cregs}
        for bits, count in counts.items():
            # Maestro returns classical bit 0 first, including ancilla registers.
            if len(bits) != qc.num_clbits or set(bits) - {"0", "1"}:
                raise ValueError(f"unexpected Maestro classical outcome: {bits!r}")
            for reg in qc.cregs:
                key = "".join(bits[qc.find_bit(bit).index] for bit in reversed(reg))
                registers[reg.name][key] = registers[reg.name].get(key, 0) + int(count)
        out.append(registers)
    return out


class MaestroBackend(Backend):
    """Maestro adapter for the standard plan/submit/harvest workflow.

    Accepts the same graph, circuit dialect and validation rules as AerBackend.
    Pass a native ``SimulatorConfig`` to select an engine or tune MPS truncation.
    Counts are stored in the manifest, so harvesting needs no running simulator.
    """

    vendor = "maestro"
    is_simulator = True
    max_circuits_per_job = 10 ** 9
    kinds = ("mcm", "direct")

    def __init__(self, graph=None, name="maestro_simulator", dialect="dynamic",
                 config=None, seed=None, rules=None, gpu: bool = False):
        super().__init__(name)
        if dialect not in ("dynamic", "iqm"):
            raise ValueError("dialect must be 'dynamic' or 'iqm'")
        self.graph = graph if graph is not None else nx.path_graph(9)
        self.dialect = dialect
        self.seed = seed
        self.rules = rules
        self.frames = ("Z", "X", "XZ") if dialect == "dynamic" else ("Z",)
        self._submitted = 0
        self.gpu = gpu
        if gpu:
            try:
                import maestro
                maestro.init_gpu()
                if config is None:
                    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.Gpu)
                else:
                    config = copy.copy(config)
                    config.simulator_type = maestro.SimulatorType.Gpu
            except Exception:
                pass
        self.config = config

    def coupling_graph(self):
        return self.graph

    def feedforward_groups(self):
        return self.rules.feedforward_groups() if self.rules is not None else None

    def validate(self, batch, kind="mcm"):
        if self.rules is not None:
            return self.rules.validate(batch, kind)
        return super().validate(batch, kind)

    def build(self, batch, depth, delta, kind="mcm", frame="Z", readout=None):
        if self.dialect == "iqm":
            if frame != "Z" or readout not in (None, "Z"):
                raise ValueError("the IQM dialect is built and read in the Z frame only")
            return iqm_to_dynamic(build_iqm(batch, depth, delta, kind))
        return build_dynamic(batch, depth, delta, kind, frame=frame, readout=readout)

    def submit(self, circuits, shots):
        circuits = list(circuits)
        base_seed = self.seed if self.seed is not None else getattr(self.config, "seed", None)
        seed = None if base_seed is None else base_seed + self._submitted
        results = run_maestro(circuits, shots, config=self.config, seed=seed)
        counts = [[reverse_keys(regs[reg.name]) for reg in qc.cregs if reg.name.startswith("d")]
                  for qc, regs in zip(circuits, results)]
        self._submitted += len(circuits)
        return {"job_id": local_job_id(self.name), "counts": counts}

    def fetch(self, record, tasks):
        return stored_counts(record, tasks)

    def estimate_observables(self, circuit, observables: list[str]):
        return estimate_maestro(circuit, observables, config=self.config)

    def ideal_energy(self, code, depth: int, delta: float = 0.5, use_subspace: bool = True,
                     ladder: bool = False, order: list[int] | None = None) -> float:
        return maestro_energy(code, depth, delta, config=self.config, use_subspace=use_subspace,
                              ladder=ladder, order=order)



