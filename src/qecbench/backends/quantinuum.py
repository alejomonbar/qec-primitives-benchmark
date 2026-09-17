"""Quantinuum Helios through Nexus (``qnexus``), with circuits written in Guppy.

Helios is all-to-all connected and reuses qubits: an ancilla is allocated, measured mid-circuit
and freed inside the program. There is no physical layout to choose, so an instance is a
*logical* chain: data qubits ``0..n-1`` (bit ``i`` of a result is data qubit ``i``) and, for
the ``mcm`` kind, ancilla labels ``n..2n-2``, one per bond. The labels name the gadgets. Which
physical ions run them is the machine's business. ``chain_instances`` builds them.

Circuits, one program per chain length and depth:

``mcm``     ``qaoa_mcm_parallel``. The gadgets of one layer run in two colour classes, the even
            bonds ``(2k, 2k+1)`` and then the odd bonds ``(2k+1, 2k+2)``. Within a class they
            touch disjoint data qubits, so none waits on another: two logical rounds per layer
            instead of a dependency chain of ``n-1`` gadgets. Physically, at most as many gadgets
            run at once as the machine has operation zones (8 on Helios-1), so a class of ``k``
            takes about ``ceil(k/8)`` steps. Each class allocates its ancillas together, runs
            ``CX CX RZ(2 gamma) H`` on each, reads them with one ``measure_array`` and applies
            ``Z Z`` where the outcome is 1. The corrections commute, so this is the same operation
            as the gadgets one by one. A class holds at most ``ceil((n-1)/2)`` ancillas and is
            measured (freeing them) before the next is allocated, so a program needs
            ``n + ceil((n-1)/2)`` qubits at once: the zones limit concurrency, not memory.
``direct``  ``CX RZ(2 gamma) CX`` on every bond, no ancilla and no measurement until the end.

Both end with ``RX(-2 beta)`` per layer and ``result("c", ...)`` holding the measured data bits.
The programs need guppylang >= 1.1 (the ``quantinuum`` extra).

``local=True`` runs the plan on Aer instead of Nexus, needing no account. It runs a Qiskit copy of
each program, gate for gate and in the same order: for ``mcm`` the two classes with ancillas reset
and reused, for ``direct`` ``CX RZ CX`` bond by bond.
Guppy's own emulator is not used, because its build step depends on the local toolchain. For a
Quantinuum-side simulation use ``Helios-1E``, the hosted emulator with its noise model. Both
flag their results as simulated.

Cost is in HQCs, predicted by Nexus (``quote``) for the uploaded programs. There is no offline
formula for Helios, so ``estimate`` returns nothing until a quote exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from uuid import uuid4

import networkx as nx

from ..circuits import batch_kind
from ..lrqaoa import angles
from ..primitives import Chain, Direct
from .aer import local_job_id, simulate_job
from .base import Backend

SYSTEMS = {                       # qubits available to one program
    "Helios-1": {"qubits": 98, "emulator": False},
    "Helios-1E": {"qubits": 98, "emulator": True},
}


def chain_instances(n_data, kinds=("mcm", "direct")):
    """Logical chain instances for every length in ``n_data``: ``Chain`` for ``mcm``, ``Direct``
    for ``direct``, labelled as described in the module docstring."""
    out = []
    for n in ([n_data] if isinstance(n_data, int) else n_data):
        if "mcm" in kinds:
            out.append(Chain.from_roles(range(n), range(n, 2 * n - 1)))
        if "direct" in kinds:
            out.append(Direct(range(n)))
    return out


def peak_qubits(n: int, kind: str) -> int:
    """Qubits a program holds at once: data plus the larger colour class of ancillas."""
    return n + ceil((n - 1) / 2) if kind == "mcm" else n


@dataclass
class HeliosProgram:
    """A Guppy program and what the backend needs to send it."""

    definition: object
    n: int
    kind: str
    depth: int
    name: str = field(default_factory=lambda: f"qecbench-{uuid4().hex[:8]}")

    @property
    def peak_qubits(self):
        return peak_qubits(self.n, self.kind)


# --------------------------------------------------------------------------------------
# Guppy programs
# --------------------------------------------------------------------------------------
def guppy_mcm_parallel(num_data_qubits: int, depth: int, delta: float = 0.5):
    """1D-chain LR-QAOA with the MCM gadgets in two colour classes (``benchmarking_quantinuum``).

    Index arithmetic rather than a comptime list of edges: Guppy cannot index a comptime list
    with a runtime loop variable. Guppy angles are in half-turns, hence the division by pi.
    Written for guppylang >= 1.1, where measurements are lazy: ``collect_measurements`` turns a
    class's outcomes into bools once they are needed for the corrections.
    """
    import numpy as np
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import array, comptime, result
    from guppylang.std.quantum import collect_measurements, cx, h, measure_array, qubit, rx, rz, z

    gammas, betas = angles(depth, delta)
    num_layers = len(gammas)
    pi_num = float(np.pi)
    n_even = len(range(0, num_data_qubits - 1, 2))   # bonds (0,1), (2,3), ...
    n_odd = len(range(1, num_data_qubits - 1, 2))    # bonds (1,2), (3,4), ...

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(comptime(num_data_qubits)))
        for i in range(comptime(num_data_qubits)):
            h(qs[i])
        for layer_i in range(comptime(num_layers)):
            anc0 = array(qubit() for _ in range(comptime(n_even)))
            for k in range(comptime(n_even)):
                cx(qs[2 * k], anc0[k])
                cx(qs[2 * k + 1], anc0[k])
                rz(anc0[k], angle(2 * comptime(gammas)[layer_i] / comptime(pi_num)))
                h(anc0[k])
            m0 = collect_measurements(measure_array(anc0))
            for k in range(comptime(n_even)):
                if m0[k]:
                    z(qs[2 * k])
                    z(qs[2 * k + 1])
            anc1 = array(qubit() for _ in range(comptime(n_odd)))
            for k in range(comptime(n_odd)):
                cx(qs[2 * k + 1], anc1[k])
                cx(qs[2 * k + 2], anc1[k])
                rz(anc1[k], angle(2 * comptime(gammas)[layer_i] / comptime(pi_num)))
                h(anc1[k])
            m1 = collect_measurements(measure_array(anc1))
            for k in range(comptime(n_odd)):
                if m1[k]:
                    z(qs[2 * k + 1])
                    z(qs[2 * k + 2])
            for i in range(comptime(num_data_qubits)):
                rx(qs[i], angle(-2.0 * comptime(betas)[layer_i] / comptime(pi_num)))
        result("c", collect_measurements(measure_array(qs)))

    return main


def guppy_direct(num_data_qubits: int, depth: int, delta: float = 0.5):
    """1D-chain LR-QAOA with ``CX RZ(2 gamma) CX`` on every bond (``qaoa_normal``)."""
    import numpy as np
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import array, comptime, result
    from guppylang.std.quantum import collect_measurements, cx, h, measure_array, qubit, rx, rz

    gammas, betas = angles(depth, delta)
    num_layers = len(gammas)
    pi_num = float(np.pi)
    n_bonds = num_data_qubits - 1

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(comptime(num_data_qubits)))
        for i in range(comptime(num_data_qubits)):
            h(qs[i])
        for layer_i in range(comptime(num_layers)):
            for i in range(comptime(n_bonds)):
                cx(qs[i], qs[i + 1])
                rz(qs[i + 1], angle(2.0 * comptime(gammas)[layer_i] / comptime(pi_num)))
                cx(qs[i], qs[i + 1])
            for i in range(comptime(num_data_qubits)):
                rx(qs[i], angle(-2.0 * comptime(betas)[layer_i] / comptime(pi_num)))
        result("c", collect_measurements(measure_array(qs)))

    return main


# --------------------------------------------------------------------------------------
# The same program in Qiskit, for local runs
# --------------------------------------------------------------------------------------
def qiskit_mcm_parallel(num_data_qubits: int, depth: int, delta: float = 0.5):
    """``guppy_mcm_parallel`` gate for gate: two colour classes, ancillas reset and reused.

    Qubits ``0..n-1`` are data, the next ``ceil((n-1)/2)`` the reusable ancillas. Register ``d0``
    holds the data bits (bit ``j`` = data qubit ``j``), ``a0`` the latest outcome of every bond.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    n = num_data_qubits
    gammas, betas = angles(depth, delta)
    n_anc = ceil((n - 1) / 2)
    anc_reg, data_reg = ClassicalRegister(n - 1, "a0"), ClassicalRegister(n, "d0")
    qc = QuantumCircuit(QuantumRegister(n + n_anc, "q"), anc_reg, data_reg)
    anc = list(range(n, n + n_anc))
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for first in (0, 1):                                   # even bonds, then odd bonds
            bonds = list(range(first, n - 1, 2))
            qc.reset(anc[:len(bonds)])
            for k, b in enumerate(bonds):
                qc.cx(b, anc[k])
                qc.cx(b + 1, anc[k])
                qc.rz(2 * gamma, anc[k])
                qc.h(anc[k])
            for k, b in enumerate(bonds):
                qc.measure(anc[k], anc_reg[b])
            for k, b in enumerate(bonds):
                with qc.if_test((anc_reg[b], 1)):
                    qc.z(b)
                    qc.z(b + 1)
        qc.rx(-2 * beta, range(n))
    qc.measure(range(n), data_reg)
    return qc


def qiskit_direct(num_data_qubits: int, depth: int, delta: float = 0.5):
    """``guppy_direct`` gate for gate: ``CX RZ(2 gamma) CX`` bond by bond, in index order.

    Register ``d0`` holds the data bits (bit ``j`` = data qubit ``j``).
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    n = num_data_qubits
    gammas, betas = angles(depth, delta)
    data_reg = ClassicalRegister(n, "d0")
    qc = QuantumCircuit(QuantumRegister(n, "q"), data_reg)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * gamma, i + 1)
            qc.cx(i, i + 1)
        qc.rx(-2 * beta, range(n))
    qc.measure(range(n), data_reg)
    return qc


# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------
class QuantinuumBackend(Backend):
    vendor = "quantinuum"
    kinds = ("mcm", "direct")
    max_circuits_per_job = 16

    def __init__(self, name="Helios-1", local=False, project="Helios-Samples", noise_model=None,
                 seed=None, cost_margin=3.0):
        """``name`` is ``Helios-1`` (hardware) or ``Helios-1E`` (Quantinuum's hosted emulator).

        ``local=True`` runs on Aer as ``<name>_sim`` (noiseless unless ``noise_model`` is given).
        ``project`` is the Nexus project jobs are filed under, and ``cost_margin`` the HQCs added
        on top of the Nexus prediction for a job's ``max_cost``.
        """
        if name not in SYSTEMS:
            raise ValueError(f"unknown Quantinuum system {name!r}; have {list(SYSTEMS)}")
        super().__init__(name + ("_sim" if local else ""))
        self.system_name = name
        self.qubits = SYSTEMS[name]["qubits"]
        self.is_simulator = SYSTEMS[name]["emulator"]
        self.local = local
        self.project = project
        self.noise_model = noise_model
        self.seed = seed
        self.cost_margin = cost_margin
        self._submitted = 0
        self._uploads = {}           # program name -> HUGRRef
        self._quotes = {}            # program name -> (predicted HQC, confidence)
        self._connected = False

    @property
    def noise_description(self):
        if self.local:
            return ("Aer, noiseless" if self.noise_model is None else
                    "Aer, qiskit_aer noise model supplied by the caller") + ", Qiskit copy of the Guppy program"
        if self.is_simulator:
            return f"{self.system_name}: Quantinuum's hosted emulator and its Helios noise model"
        return None

    def coupling_graph(self):
        """All-to-all. Labels are logical and go up to ``2 * qubits``, the most a chain can name."""
        return nx.complete_graph(2 * self.qubits)

    # -- circuits -------------------------------------------------------------------------
    def validate(self, batch, kind="mcm"):
        problems = []
        if len(batch) != 1:
            problems.append(f"one chain per program on Helios; got {len(batch)} in one batch")
        for inst in batch:
            n = inst.n_data
            if inst.kind != kind:
                problems.append(f"{inst}: kind {inst.kind}, batch kind {kind}")
            expected = chain_instances(n, kinds=(kind,))[0]
            if inst != expected:
                problems.append(f"{inst}: Helios chains use logical labels, data 0..{n - 1}"
                                + (f" and ancillas {n}..{2 * n - 2}" if kind == "mcm" else "")
                                + " (build them with chain_instances)")
            if peak_qubits(n, kind) > self.qubits:
                problems.append(f"{inst}: needs {peak_qubits(n, kind)} qubits at once, "
                                f"{self.system_name} has {self.qubits}")
        return problems

    def build(self, batch, depth, delta, kind="mcm"):
        kind = batch_kind(batch, kind)
        n = batch[0].n_data
        if self.local:
            return (qiskit_mcm_parallel if kind == "mcm" else qiskit_direct)(n, depth, delta)
        make = guppy_mcm_parallel if kind == "mcm" else guppy_direct
        return HeliosProgram(make(n, depth, delta), n, kind, depth,
                             name=f"qecbench-chain{n}-{kind}-p{depth}-{uuid4().hex[:8]}")

    # -- Nexus ------------------------------------------------------------------------------
    def _qnx(self):
        import qnexus as qnx

        if not self._connected:
            project = qnx.projects.get_or_create(self.project)
            qnx.context.set_active_project(project)
            self._connected = True
        return qnx

    def _upload(self, programs):
        qnx = self._qnx()
        for prog in programs:
            if prog.name not in self._uploads:
                self._uploads[prog.name] = qnx.hugr.upload(prog.definition.compile(), name=prog.name)
        return [self._uploads[prog.name] for prog in programs]

    def quote(self, plan):
        """Upload the plan's programs and ask Nexus for their predicted HQC cost.

        Uploading costs nothing; the prediction is stored in ``plan["estimate"]`` (so
        ``print_plan`` shows it) and reused for each job's ``max_cost`` at submission.
        """
        if self.local:
            raise ValueError("a local backend has no Nexus quote")
        programs = plan["circuits"]
        refs = self._upload(programs)
        predicted = self._qnx().hugr.cost_confidence(programs=refs, n_shots=[plan["shots"]] * len(refs),
                                                     system_name="Helios-1")
        for prog, (cost, confidence) in zip(programs, predicted):
            self._quotes[prog.name] = (float(cost), float(confidence))
        per_circuit = [self._quotes[p.name][0] for p in programs]
        plan["estimate"] = {"unit": "HQC", "per_circuit": per_circuit, "total": float(sum(per_circuit)),
                            "usd": None,
                            "notes": ["predicted by Nexus (qnx.hugr.cost_confidence) for the uploaded programs",
                                      f"each job's max_cost = ceil(predicted) + {self.cost_margin:g} HQC"]}
        return plan["estimate"]

    def estimate(self, plan):
        return None             # only Nexus can price a Helios program; see quote()

    # -- execution ------------------------------------------------------------------------
    def submit(self, circuits, shots):
        if self.local:
            seed = None if self.seed is None else self.seed + self._submitted
            self._submitted += len(circuits)
            return simulate_job(circuits, shots, self.noise_model, seed, job_id=local_job_id(self.name))
        qnx = self._qnx()
        refs = self._upload(circuits)
        missing = [p for p in circuits if p.name not in self._quotes]
        if missing:
            predicted = qnx.hugr.cost_confidence(programs=[self._uploads[p.name] for p in missing],
                                                 n_shots=[shots] * len(missing), system_name="Helios-1")
            for prog, (cost, confidence) in zip(missing, predicted):
                self._quotes[prog.name] = (float(cost), float(confidence))
        max_cost = float(ceil(sum(self._quotes[p.name][0] for p in circuits)) + self.cost_margin)
        options = {"system_name": self.system_name, "max_cost": max_cost}
        if self.is_simulator:
            options["emulator_config"] = qnx.models.HeliosEmulatorConfig(
                n_qubits=max(p.peak_qubits for p in circuits))
        job = qnx.start_execute_job(
            programs=refs, n_shots=[shots] * len(refs), backend_config=qnx.models.HeliosConfig(**options),
            name=f"qecbench-{self.system_name}-{len(refs)}programs-{uuid4().hex[:8]}")
        return {"job_id": str(job.id), "programs": [p.name for p in circuits], "max_cost": max_cost}

    def fetch(self, record, tasks):
        if "counts" in record:
            return [record["counts"][t["index"]] for t in tasks]
        qnx = self._qnx()
        ref = qnx.jobs.get(id=record["job_id"])
        status = qnx.jobs.status(ref)
        if str(getattr(status.status, "value", status.status)) != "COMPLETED":
            print(f"  {record['job_id']}: {getattr(status.status, 'value', status.status)}"
                  + (f" - {status.message}" if getattr(status, "message", None) else ""))
            return None
        items = sorted(qnx.jobs.results(ref), key=lambda item: item.job_item_integer_id)
        if len(items) != len(record.get("programs", items)):
            raise ValueError(f"job {record['job_id']}: {len(items)} results for "
                             f"{len(record['programs'])} programs")
        return [[bitstring_counts(items[t["index"]].download_result())] for t in tasks]


def bitstring_counts(result, register="c"):
    """``{bitstring: count}`` of a Helios result; character ``i`` is data qubit ``i``."""
    counts = {}
    for shot in getattr(result, "results", result):
        bits = shot.to_register_bits()[register]
        counts[bits] = counts.get(bits, 0) + 1
    return counts
