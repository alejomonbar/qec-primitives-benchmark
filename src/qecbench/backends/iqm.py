"""IQM Garnet / Emerald through Amazon Braket, driven from Qiskit (qiskit-braket-provider).

The circuit is built directly in IQM natives (``build_iqm``) and submitted verbatim inside
``EnableExperimentalCapability`` - the provider then emits ``measure_ff`` / ``cc_prx`` and
Braket flags the task as experimental.  Rules enforced by ``validate`` before anything is
sent (AWS developer guide, "Dynamic circuits on IQM devices"):

1. feedback keys are unique and a ``cc_prx`` follows its ``measure_ff`` (by construction);
2. **one controller per qubit per circuit** - a qubit's feed-forward may be controlled by a
   single qubit (itself or another);
3. control only inside a **feed-forward group** (``FF_GROUPS``);
4. verbatim submission (by construction).

Rule 2 is what limits chains: in ``d0-a0-d1-a1-d2`` the middle data qubit needs corrections
from both ``a0`` and ``a1``.  So on IQM the MCM gadget runs on triplets (``n_data = 2``);
longer chains are rejected with an explanation.  ``Direct`` instances (kind ``direct``) have
no feed-forward at all and run at any length.

Billing is per task + per shot, and every circuit is its own task.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..analysis import short_path
from ..circuits import build_iqm
from ..layout import load_cached_graph, save_graph, validate_batch
from .aer import local_job_id, simulate_job
from .base import Backend

PRICING = {  # USD, aws.amazon.com/braket/pricing, checked 2026-08-12
    "iqm_garnet": {"per_task": 0.30, "per_shot": 0.00145},
    "iqm_emerald": {"per_task": 0.30, "per_shot": 0.00160},
}

DEVICE_NAMES = {"iqm_garnet": "Garnet", "iqm_emerald": "Emerald"}

# Transcribed from the AWS qubit-grouping figure for dynamic circuits.
FF_GROUPS = {
    "iqm_garnet": {
        1: [8, 13, 14, 15, 18, 19],
        2: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 16, 17, 20],
    },
    "iqm_emerald": {
        1: [3, 4, 8, 9, 10, 18, 31, 39, 46, 51, 53, 54],
        2: [1, 2, 5, 6, 7, 11, 12, 13, 14, 19, 20, 21, 27, 28],
        3: [15, 16, 17, 22, 23, 24, 25, 26, 29, 30, 36, 37, 38, 45],
        4: [32, 33, 34, 35, 40, 41, 42, 43, 44, 47, 48, 49, 50, 52],
    },
}


def validate_iqm_feedforward(batch, groups=None) -> list[str]:
    """Rules 2 and 3 above for the ``mcm`` kind."""
    grp = {q: g for g, qs in (groups or {}).items() for q in qs}
    problems, controller = [], {}
    for inst in batch:
        for term in inst.terms:
            if grp:
                gs = {q: grp.get(q) for q in (term.ancilla, *term.data)}
                if None in gs.values() or len(set(gs.values())) > 1:
                    problems.append(f"{inst}: term on ancilla {term.ancilla} spans feed-forward "
                                    f"groups {gs}")
            for target in (term.ancilla, *term.data):
                if controller.setdefault(target, term.ancilla) != term.ancilla:
                    problems.append(
                        f"{inst}: qubit {target} would be feed-forward controlled by both "
                        f"{controller[target]} and {term.ancilla} (IQM allows one controller "
                        f"per qubit per circuit - MCM chains need n_data = 2 on IQM)")
    return problems


class IQMBackend(Backend):
    vendor = "iqm"
    kinds = ("mcm", "direct")
    max_circuits_per_job = 1

    def __init__(self, name="iqm_garnet", backend=None, local=False, noise_model=None, seed=None):
        """``local=True`` runs the plan on Aer instead of AWS (the IQM circuit rewritten to
        ``if_test``), needing no credentials.  It is noiseless unless a ``noise_model`` is
        supplied, since Braket publishes fidelities but no error model for IQM.  The backend
        then calls itself ``<device>_sim`` so its files never mix with real ones."""
        if name not in DEVICE_NAMES:
            raise ValueError(f"unknown IQM device {name!r}; have {list(DEVICE_NAMES)}")
        super().__init__(name + ("_sim" if local else ""))
        self.device_name = name
        self._backend = backend
        self.local = local
        self.noise_model = noise_model
        self.seed = seed
        self._submitted = 0
        self._graph = None

    @property
    def backend(self):
        """The ``BraketAwsBackend`` (created on first use, so planning needs no credentials)."""
        if self._backend is None:
            from qiskit_braket_provider import BraketProvider

            self._backend = BraketProvider().get_backend(DEVICE_NAMES[self.device_name])
        return self._backend

    # -- device ---------------------------------------------------------------------------
    def coupling_graph(self, refresh=False):
        if self._graph is None or refresh:
            G = None if refresh else load_cached_graph(self.device_name)
            if G is None:
                import networkx as nx

                topo = self.backend._device.topology_graph
                G = nx.Graph()
                G.add_edges_from((int(u), int(v)) for u, v in topo.edges)
                save_graph(self.device_name, G)
            self._graph = G
        return self._graph

    def qubit_labels(self):
        """Device labels in the order the provider maps circuit indices onto them."""
        if self._backend is not None and getattr(self._backend, "qubit_labels", None):
            return tuple(self._backend.qubit_labels)
        return tuple(sorted(self.coupling_graph().nodes))

    def feedforward_groups(self):
        return FF_GROUPS.get(self.device_name)

    def calibration(self, save_dir="data/calibration"):
        """Vendor calibration (fRO, 1Q RB, fCZ, T1/T2).  Braket publishes no durations for IQM."""
        props = self.backend._device.properties
        provider = props.provider.dict()["properties"]
        cal = {"qpu": self.device_name, "updated_at": str(props.service.updatedAt),
               "fetched_at": datetime.now().isoformat(),
               "one_qubit": {int(q): v for q, v in provider["one_qubit"].items()},
               "two_qubit": dict(provider["two_qubit"])}
        if save_dir:
            path = (Path(save_dir) / self.name /
                    f"{props.service.updatedAt:%Y%m%d_%H%M}_{self.name}_calibration.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cal, indent=1))
            print(f"calibration of {cal['updated_at']} saved to {short_path(path)}")
        return cal

    def error_budget(self, instance, cal):
        one, two = cal["one_qubit"], cal["two_qubit"]
        q = lambda x: one.get(x) or one.get(str(x)) or {}

        def fcz(u, v):
            e = two.get(f"{u}-{v}") or two.get(f"{v}-{u}")
            return 1.0 if not e else e.get("fCZ", 1.0)

        eps = sum(1 - fcz(u, v) for u, v in instance.couplers)
        eps += sum(1 - q(a).get("fRO", 1.0) for a in instance.ancillas)
        eps += sum(1 - q(x).get("f1Q_simultaneous_RB", 1.0) for x in instance.qubits)
        return eps

    def flag_instances(self, instances, cal, min_fcz=0.95, min_fro=0.90):
        """Instances on a coupler or readout worse than the thresholds, and why.

        Braket publishes fidelities rather than errors for IQM, and no mid-circuit-specific
        readout figure, so the terminal ``fRO`` stands in for the ancilla too.
        """
        one, two = cal["one_qubit"], cal["two_qubit"]
        q = lambda x: one.get(x) or one.get(str(x)) or {}
        show = lambda f: "not calibrated" if f is None else f"{f:.3f}"

        def fcz(u, v):
            entry = two.get(f"{u}-{v}") or two.get(f"{v}-{u}")
            return None if not entry else entry.get("fCZ")

        flagged = []
        for inst in instances:
            reasons = []
            for u, v in inst.couplers:
                f = fcz(u, v)
                if f is None or f < min_fcz:
                    reasons.append(f"coupler {u}-{v} fCZ {show(f)}")
            for x in inst.qubits:
                f = q(x).get("fRO")
                if f is None or f < min_fro:
                    reasons.append(f"qubit {x} fRO {show(f)}")
            if reasons:
                flagged.append((inst, "; ".join(reasons[:3])))
        return flagged

    # -- circuits -------------------------------------------------------------------------
    def validate(self, batch, kind="mcm"):
        problems = validate_batch(batch, self.coupling_graph())
        if kind == "mcm":
            problems += validate_iqm_feedforward(batch, self.feedforward_groups())
        return problems

    def build(self, batch, depth, delta, kind="mcm"):
        labels = self.qubit_labels()
        return build_iqm(batch, depth, delta, kind,
                         qubit_index={q: i for i, q in enumerate(labels)}, num_qubits=len(labels))

    def to_braket(self, circuit):
        """The Braket program that will actually be sent - for inspection."""
        from braket.experimental_capabilities import EnableExperimentalCapability
        from qiskit_braket_provider.providers.adapter import to_braket

        with EnableExperimentalCapability():
            return to_braket(circuit, qubit_labels=self.qubit_labels(), verbatim=True)

    def estimate(self, plan):
        price = PRICING[self.device_name]
        per_circuit = [price["per_task"] + plan["shots"] * price["per_shot"] for _ in plan["circuits"]]
        total = sum(per_circuit)
        return {"unit": "USD", "per_circuit": per_circuit, "total": total, "usd": total,
                "notes": [f"${price['per_task']:.2f}/task + ${price['per_shot']:.5f}/shot, "
                          "one task per circuit; Braket credits 1:1 with USD"]}

    # -- execution ------------------------------------------------------------------------
    def submit(self, circuits, shots):
        if len(circuits) != 1:
            raise ValueError("IQM: one circuit per task")
        if self.local:
            from ..circuits import iqm_to_dynamic

            seed = None if self.seed is None else self.seed + self._submitted
            self._submitted += 1
            return simulate_job([iqm_to_dynamic(circuits[0])], shots, self.noise_model, seed,
                                job_id=local_job_id(self.name))
        from braket.experimental_capabilities import EnableExperimentalCapability

        with EnableExperimentalCapability():
            task = self.backend.run(circuits[0], shots=shots, verbatim=True)
        return {"job_id": task.job_id()}

    def fetch(self, record, tasks):
        if "counts" in record:
            return [record["counts"][t["index"]] for t in tasks]
        from braket.aws import AwsQuantumTask

        from ..primitives import from_dict

        task = AwsQuantumTask(arn=record["job_id"])
        state = task.state()
        if state != "COMPLETED":
            print(f"  {record['job_id']}: {state}")
            return None
        result = task.result()
        order = [int(q) for q in result.measured_qubits]
        counts = dict(result.measurement_counts)
        out = []
        for t in tasks:
            per_inst = []
            for spec in t["instances"]:
                idx = [order.index(q) for q in from_dict(spec).data_qubits]
                marg = {}
                for bits, n in counts.items():
                    key = "".join(bits[i] for i in idx)
                    marg[key] = marg.get(key, 0) + int(n)
                per_inst.append(marg)
            out.append(per_inst)
        return out
