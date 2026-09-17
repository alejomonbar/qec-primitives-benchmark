"""IBM Quantum (Qiskit Runtime) adapter.

What drives the design:

* **No feed-forward groups** - any coupled path is executable, so chains of any length run.
* **Billing is time.**  The estimate schedules every transpiled circuit against the
  target's own durations; the ancilla is charged ``measure_2`` (IBM's *mid-circuit*
  readout, published in ``backend.properties()`` and never in the target), the data qubits
  the terminal ``measure``.
* **Feed-forward has no published duration.**  Each conditional is charged
  ``FEEDFORWARD_PER_CONDITIONAL`` - a HYPOTHESIS inferred from billed seconds on
  ibm_kingston (2026-08-13) and not confirmed since.
  Treat every "QPU s" as an order of magnitude.
* **Classical control memory** (error 6073) caps circuits per job: ``max_circuits_per_job``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import networkx as nx

from ..circuits import build_dynamic, compact_index
from .base import Backend, reverse_keys

USD_PER_SECOND = 1.60                 # Pay-As-You-Go, checked 2026-08-13
FEEDFORWARD_PER_CONDITIONAL = 1.9e-6  # s per conditional block per layer - unconfirmed
UNIT_SECONDS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "": 1e-9}


class IBMBackend(Backend):
    vendor = "ibm"
    kinds = ("mcm", "direct")

    def __init__(self, name=None, backend=None, account=None, service=None,
                 optimization_level=1, dynamical_decoupling=False, max_circuits_per_job=8,
                 local=False, seed=None):
        """``local=True`` runs the plan on Aer instead of submitting it.

        The noise model is built from **this device's current calibration**
        (``NoiseModel.from_backend``), and the circuits are the same transpiled, physically
        laid out ones that would have been sent - so it previews what the device is likely to
        return, using the same code path.  The backend then calls itself ``<device>_sim``, so
        its manifests, calibration snapshots and result files never mix with real ones.
        """
        self.local, self.seed, self._submitted, self._noise = local, seed, 0, None
        if backend is None:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = service or (QiskitRuntimeService(name=account) if account else QiskitRuntimeService())
            backend = service.backend(name)
        super().__init__(backend.name + ("_sim" if local else ""))
        self.backend = backend
        self.service = service or getattr(backend, "service", None)
        self.optimization_level = optimization_level
        self.dynamical_decoupling = dynamical_decoupling
        self.max_circuits_per_job = max_circuits_per_job
        self._graph = None

    # -- device ---------------------------------------------------------------------------
    def coupling_graph(self):
        if self._graph is None:
            G = nx.Graph()
            G.add_edges_from({tuple(sorted(e)) for e in self.backend.coupling_map.get_edges()})
            self._graph = G
        return self._graph

    def supports_mcm(self):
        return "measure_2" in self.backend.configuration().supported_instructions

    def measure_2(self) -> dict:
        """``{qubit: {"error", "duration"}}`` of the mid-circuit readout (empty if none)."""
        try:
            props = self.backend.properties()
        except Exception:
            return {}
        out = {}
        for gate in getattr(props, "gates", []):
            if gate.gate != "measure_2":
                continue
            p = {x.name: x for x in gate.parameters}
            length, error = p.get("gate_length"), p.get("gate_error")
            out[gate.qubits[0]] = {
                "error": None if error is None else error.value,
                "duration": None if length is None else length.value * UNIT_SECONDS.get(length.unit, 1e-9)}
        return out

    def calibration(self, save_dir="data/calibration"):
        """Readout (terminal and mid-circuit), CZ, sx, T1/T2 snapshot - timestamped on disk."""
        target = self.backend.target
        ops = target.operation_names
        two_q = next((g for g in ("cz", "ecr", "cx") if g in ops), None)
        mcm = self.measure_2()
        one = {}
        for q in range(self.backend.num_qubits):
            meas = target["measure"].get((q,)) if "measure" in ops else None
            sx = target["sx"].get((q,)) if "sx" in ops else None
            qp = self.backend.qubit_properties(q)
            one[q] = {"readout_error": getattr(meas, "error", None),
                      "readout_duration": getattr(meas, "duration", None),
                      "mcm_readout_error": mcm.get(q, {}).get("error"),
                      "mcm_readout_duration": mcm.get(q, {}).get("duration"),
                      "sx_error": getattr(sx, "error", None),
                      "T1": getattr(qp, "t1", None), "T2": getattr(qp, "t2", None)}
        two = {f"{u}-{v}": {"error": getattr(p, "error", None), "duration": getattr(p, "duration", None)}
               for (u, v), p in (target[two_q].items() if two_q else [])}
        cal = {"backend": self.name, "fetched_at": datetime.now().isoformat(),
               "two_qubit_gate": two_q, "one_qubit": one, "two_qubit": two}
        if save_dir:
            path = (Path(save_dir) / self.name /
                    f"{datetime.now():%Y%m%d_%H%M}_{self.name}_calibration.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cal, indent=1))
            print(f"calibration saved to {path}")
        return cal

    def error_budget(self, instance, cal):
        """Per layer: CZ error of every coupler + ``measure_2`` of every ancilla + 2 sx per qubit."""
        two, one = cal["two_qubit"], cal["one_qubit"]
        q = lambda x: one.get(x) or one.get(str(x)) or {}

        def cz(u, v):
            e = two.get(f"{u}-{v}") or two.get(f"{v}-{u}")
            return 0.0 if not e or e["error"] is None else e["error"]

        eps = sum(cz(u, v) for u, v in instance.couplers)
        eps += sum(q(a).get("mcm_readout_error") or q(a).get("readout_error") or 0.0
                   for a in instance.ancillas)
        eps += sum(2 * (q(x).get("sx_error") or 0.0) for x in instance.qubits)
        return eps

    def flag_instances(self, instances, cal, max_2q_error=0.05, max_readout_error=0.1):
        """Instances touching a coupler or a readout worse than the thresholds, and why.

        The ancilla is judged on ``measure_2`` (it is read mid-circuit), the data qubits on
        the terminal ``measure``.  A missing number counts as bad: it usually means the gate
        or qubit failed calibration.
        """
        two, one = cal["two_qubit"], cal["one_qubit"]
        q = lambda x: one.get(x) or one.get(str(x)) or {}
        show = lambda e: "not calibrated" if e is None else f"{e:.3f}"

        def cz(u, v):
            entry = two.get(f"{u}-{v}") or two.get(f"{v}-{u}")
            return None if not entry else entry["error"]

        flagged = []
        for inst in instances:
            reasons = []
            for u, v in inst.couplers:
                err = cz(u, v)
                if err is None or err > max_2q_error:
                    reasons.append(f"coupler {u}-{v} error {show(err)}")
            for a in inst.ancillas:
                err = q(a).get("mcm_readout_error") or q(a).get("readout_error")
                if err is None or err > max_readout_error:
                    reasons.append(f"ancilla {a} mid-circuit readout {show(err)}")
            for d in inst.data_qubits:
                err = q(d).get("readout_error")
                if err is None or err > max_readout_error:
                    reasons.append(f"qubit {d} readout {show(err)}")
            if reasons:
                flagged.append((inst, "; ".join(reasons[:3])))
        return flagged

    # -- circuits -------------------------------------------------------------------------
    def build(self, batch, depth, delta, kind="mcm"):
        from qiskit import transpile

        qc = build_dynamic(batch, depth, delta, kind)
        _, layout = compact_index(batch)
        return transpile(qc, backend=self.backend, initial_layout=layout,
                         optimization_level=self.optimization_level)

    def check_built(self, circuit, batch):
        """Two-qubit gates outside the instances' couplers mean the transpiler routed."""
        allowed = {c for inst in batch for c in inst.couplers}
        routed = set()
        for inst in circuit.data:
            op = inst.operation
            if getattr(op, "blocks", None) or op.name == "barrier" or len(inst.qubits) != 2:
                continue
            pair = tuple(sorted(circuit.find_bit(b).index for b in inst.qubits))
            if pair not in allowed:
                routed.add((op.name, pair))
        return [f"routing gate {name} on {pair}" for name, pair in sorted(routed)]

    def circuit_seconds(self, qc, mcm_durations=None):
        """ASAP schedule of a transpiled circuit against target durations (no feed-forward)."""
        target = self.backend.target
        dt = getattr(target, "dt", None) or 1.0
        last = {}
        for k, inst in enumerate(qc.data):
            for b in inst.qubits:
                last[qc.find_bit(b).index] = k
        clock = [0.0] * qc.num_qubits
        for k, inst in enumerate(qc.data):
            qs = tuple(qc.find_bit(b).index for b in inst.qubits)
            name = inst.operation.name
            if not qs or name in ("barrier", "if_else", "rz"):
                continue
            if name == "delay":
                duration = float(inst.operation.params[0]) * dt
            elif name == "measure" and mcm_durations and last[qs[0]] != k and mcm_durations.get(qs[0]):
                duration = mcm_durations[qs[0]]
            else:
                try:
                    duration = getattr(target[name][qs], "duration", 0.0) or 0.0
                except (KeyError, TypeError):
                    duration = 0.0
            stop = max(clock[q] for q in qs) + duration
            for q in qs:
                clock[q] = stop
        return max(clock, default=0.0)

    def estimate(self, plan):
        mcm = {q: v["duration"] for q, v in self.measure_2().items() if v["duration"]}
        rep_delay = getattr(self.backend.configuration(), "default_rep_delay", None) or 2.5e-4
        per_circuit = []
        for task, qc in zip(plan["tasks"], plan["circuits"]):
            seconds = self.circuit_seconds(qc, mcm)
            if task["kind"].startswith("mcm"):
                n_cond = sum(len(inst.terms) for inst in task["instances"])
                seconds += task["depth"] * n_cond * FEEDFORWARD_PER_CONDITIONAL
            per_circuit.append((seconds + rep_delay) * plan["shots"])
        total = sum(per_circuit)
        return {"unit": "QPU s", "per_circuit": per_circuit, "total": total,
                "usd": total * USD_PER_SECOND,
                "notes": [f"rep_delay {rep_delay * 1e6:.0f} us/shot; feed-forward charged "
                          f"{FEEDFORWARD_PER_CONDITIONAL * 1e6:.1f} us per conditional per layer "
                          "(unconfirmed hypothesis)"]}

    # -- execution ------------------------------------------------------------------------
    @property
    def noise_description(self):
        return (f"qiskit_aer NoiseModel.from_backend({self.backend.name}), calibration of "
                f"{datetime.now():%Y-%m-%d}") if self.local else None

    def submit(self, circuits, shots):
        if self.local:
            from qiskit_aer.noise import NoiseModel

            from .aer import local_job_id, simulate_job

            self._noise = self._noise or NoiseModel.from_backend(self.backend)
            seed = None if self.seed is None else self.seed + self._submitted
            self._submitted += len(circuits)
            return simulate_job(circuits, shots, self._noise, seed, job_id=local_job_id(self.name))
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self.backend)
        sampler.options.dynamical_decoupling.enable = self.dynamical_decoupling
        job = sampler.run(circuits, shots=shots)
        return {"job_id": job.job_id()}

    def fetch(self, record, tasks):
        if "counts" in record:
            return [record["counts"][t["index"]] for t in tasks]
        service = self.service
        if service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = self.service = QiskitRuntimeService()
        job = service.job(record["job_id"])
        status = job.status()
        if status != "DONE":
            print(f"  {record['job_id']}: {status}")
            return None
        result = job.result()
        return [[reverse_keys(getattr(result[t["index"]].data, f"d{k}").get_counts())
                 for k in range(len(t["instances"]))] for t in tasks]
