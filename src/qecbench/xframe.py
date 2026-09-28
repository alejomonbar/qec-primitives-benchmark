"""Experimental: the LR-QAOA code benchmark in the X frame.

Self-contained on purpose - nothing else imports it, so deleting this file and ``tests/test_xframe.py``
removes it without a trace.

The benchmark measures a code's checks as ``Z...Z`` terms: the data start in ``|+>``, each layer is a
measure-and-correct gadget per check followed by an ``rx`` mixer, and the data are read in ``Z``. Its **X frame**
is the same algorithm conjugated by ``H`` on every data qubit: ``X...X`` terms, an ``rz`` mixer, the data
starting in ``|0>`` and read in ``X``. Noiselessly the two give the same distribution of energies, hence the
same ``r``. On a device they differ in which physical error hurts: an error that flips the state of a data qubit
in one frame is a phase error in the other, so the X frame is to the X memory experiment what the Z frame is to
the Z memory experiment - if the frame matters, ``r_ovl`` of the X frame should rank patches the way X memory
does.

It must be built natively. Wrapping the Z-frame circuit in ``H`` layers compiles back to the same circuit (the
first ``H`` cancels the preparation, the last one the rotation to the readout basis) and would measure nothing
new. Here the data are conjugated where it changes what they physically hold: the gadgets' ``cz`` block is wrapped
as ``h(data) ... h(data)`` - so through the ancilla readout, where idle noise acts, the data sit in the X frame -
the corrections are ``X``, the mixer is ``rz``, and the data are rotated with ``H`` only for the final readout.

``build_frame`` builds either frame (the Z frame is ``circuits.build_dynamic`` gate for gate); ``simulate`` runs
both under a Pauli + relaxation model of a patch from its calibration, with the idling of the data through every
mid-circuit readout, to see whether the two frames separate by more than a device run's shot noise.
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .analysis import analyse
from .circuits import batch_kind, compact_index, cz_rounds, split_register_counts
from .lrqaoa import angles

FRAMES = ("Z", "X")


def build_frame(batch, depth, delta=0.5, frame="Z", qubit_index=None, num_qubits=None, idle=None):
    """The MCM LR-QAOA circuit of ``batch`` (``CodePatch`` instances) in the ``Z`` or ``X`` frame.

    ``idle(qc, data)``, if given, is called while the ancillas are being read, with the circuit indices of the
    data qubits - for a simulation to add the noise of that idle time.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be 'Z' or 'X', not {frame!r}")
    if batch_kind(batch) != "mcm":
        raise ValueError("the X frame is defined for the mcm (ancilla) circuit")
    if qubit_index is None:
        qubit_index, layout = compact_index(batch)
        num_qubits = len(layout)
    qi = qubit_index
    gammas, betas = angles(depth, delta)
    x = frame == "X"

    anc_regs = [ClassicalRegister(len(inst.terms), f"a{k}") for k, inst in enumerate(batch)]
    dat_regs = [ClassicalRegister(inst.n_data, f"d{k}") for k, inst in enumerate(batch)]
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"), *anc_regs, *dat_regs)
    data = [qi[d] for inst in batch for d in inst.data_qubits]
    ancs = [qi[a] for inst in batch for a in inst.ancillas]
    rounds = cz_rounds(batch)

    if not x:
        qc.h(data)                                # Z frame: |+>; X frame: H|+> = |0>, nothing to do
    for layer in range(depth):
        qc.h(ancs)
        if x:
            qc.h(data)
        for r in rounds:
            for d, a in r:
                qc.cz(qi[d], qi[a])
        if x:
            qc.h(data)
        for term in (t for inst in batch for t in inst.terms):
            qc.rx(2 * term.weight * gammas[layer], qi[term.ancilla])
        qc.barrier()
        for k, inst in enumerate(batch):
            for j, term in enumerate(inst.terms):
                qc.measure(qi[term.ancilla], anc_regs[k][j])
        if idle is not None:
            idle(qc, data)
        qc.barrier()
        for k, inst in enumerate(batch):
            for j, term in enumerate(inst.terms):
                with qc.if_test((anc_regs[k][j], 1)):
                    for d in term.data:
                        (qc.x if x else qc.z)(qi[d])
                    qc.x(qi[term.ancilla])        # ancilla back to |0> for the next layer
        qc.barrier()
        (qc.rz if x else qc.rx)(-2 * betas[layer], data)
    if x:
        qc.h(data)                                # read the data in X
    for k, inst in enumerate(batch):
        qc.measure([qi[d] for d in inst.data_qubits], dat_regs[k])
    return qc


# ======================================================================================
# Simulation: does the frame make a difference a device run could see?
# ======================================================================================
def patch_noise(patch, cal, qubit_index, idle_scale=1.0, dephasing_only=False):
    """``(NoiseModel, idle)`` for ``patch`` from a device calibration snapshot, on the circuit's indices.

    Depolarizing CZ error from each coupler's calibration, ``sx``-sized depolarizing error on ``h``/``x`` (twice
    on ``rx``; ``rz`` and ``z`` are virtual), readout error (mid-circuit for ancillas, final for data), and the idle
    time: ``idle`` puts an ``id`` on every data qubit while the ancillas are read, and the model gives that ``id``
    the Pauli-twirled thermal relaxation of the qubit for the ancilla-readout time ``t``:
    ``p_X = p_Y = (1 - exp(-t/T1)) / 4``, ``p_Z = (1 - exp(-t/T2)) / 2 - p_X``, with its own ``T1``, ``T2``. The
    twirl keeps the balance of flips against phase errors, which is what the frames differ in, and drops only the
    direction of amplitude damping. (Aer, as installed, fails with "Kraus is empty" or crashes when the exact
    relaxation channel meets the qubit-specific CZ errors in a feed-forward circuit.) ``idle_scale`` multiplies
    ``t``; ``dephasing_only`` sets ``p_X = p_Y = 0`` and keeps the phase errors.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

    one, two = cal["one_qubit"], cal["two_qubit"]
    q = lambda p: one.get(str(p)) or {}
    model = NoiseModel(basis_gates=["cz", "h", "x", "z", "rx", "rz", "id", "measure"])
    for u, v in patch.couplers:
        p = (two.get(f"{u}-{v}") or two.get(f"{v}-{u}") or {}).get("error", 1e-2)
        model.add_quantum_error(depolarizing_error(min(p, 0.75), 2), ["cz"], [qubit_index[u], qubit_index[v]])
    ancillas = set(patch.ancillas)
    for phys in patch.qubits:
        i, p1 = qubit_index[phys], q(phys).get("sx_error", 3e-4)
        model.add_quantum_error(depolarizing_error(p1, 1), ["h", "x"], [i])
        model.add_quantum_error(depolarizing_error(min(2 * p1, 0.75), 1), ["rx"], [i])
        ro = q(phys).get("mcm_readout_error" if phys in ancillas else "readout_error", 1e-2)
        model.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [i])
    duration = idle_scale * float(np.mean([q(a).get("mcm_readout_duration", 1.14e-6) for a in patch.ancillas]))
    for d in patch.data_qubits:
        t1, t2 = q(d).get("T1", 150e-6), q(d).get("T2", 150e-6)
        t2 = min(t2, 2 * t1)
        px = 0.0 if dephasing_only else (1 - np.exp(-duration / t1)) / 4
        pz = max((1 - np.exp(-duration / t2)) / 2 - (1 - np.exp(-duration / t1)) / 4, 0.0)
        model.add_quantum_error(pauli_error([("X", px), ("Y", px), ("Z", pz), ("I", 1 - 2 * px - pz)]), ["id"],
                                [qubit_index[d]])

    def idle(qc, data):
        for i in data:
            qc.id(i)
    return model, idle


def simulate(patch, cal, depths, shots=8000, delta=0.5, seed=1, **noise):
    """``{frame: {depth: analyse(...)}}`` for both frames of ``patch`` under ``patch_noise(patch, cal, **noise)``
    (``cal=None``: noiseless)."""
    from qiskit_aer import AerSimulator

    qubit_index, layout = compact_index([patch])
    model, idle = patch_noise(patch, cal, qubit_index, **noise) if cal is not None else (None, None)
    sim = AerSimulator(method="statevector", noise_model=model)
    out = {}
    for k, frame in enumerate(FRAMES):
        for j, depth in enumerate(depths):
            qc = build_frame([patch], depth, delta, frame, qubit_index, len(layout), idle=idle)
            run = sim.run(qc, shots=shots, seed_simulator=seed + 1000 * k + j)
            regs = split_register_counts(run.result().get_counts(), qc)
            counts = {bits[::-1]: n for bits, n in regs["d0"].items()}   # character i = data qubit i
            out.setdefault(frame, {})[depth] = analyse(counts, patch, depth, delta)
    return out


# ======================================================================================
# The device run: both frames interleaved, kept apart from every other result
# ======================================================================================
# Manifests under data/manifests/xframe/ and results under data/results/<backend>/surface_code/xframe/, with the
# kinds below: no other notebook or harvest looks there or at these kinds, so deleting the two folders removes
# the data as cleanly as deleting this file removes the code.
KINDS = {"Z": "xframe_z", "X": "xframe_x"}


def plan(backend, patches, depths, shots=1000, delta=0.5):
    """Every (patch, depth, frame) circuit on an ``IBMBackend``, transpiled onto the patch's own qubits exactly as
    ``IBMBackend.build`` does, the two frames of a patch and depth side by side (so they share a job).

    Refuses a circuit the transpiler routed. The estimate is ``IBMBackend.estimate``'s, as for any benchmark run.
    """
    from qiskit import transpile

    tasks, circuits = [], []
    for patch in patches:
        _, layout = compact_index([patch])
        for depth in depths:
            for frame in FRAMES:
                qc = transpile(build_frame([patch], depth, delta, frame), backend=backend.backend,
                               initial_layout=layout, optimization_level=backend.optimization_level)
                problems = backend.check_built(qc, [patch])
                if problems:
                    raise ValueError(f"{patch}, p = {depth}, {frame} frame: {problems[:3]}")
                tasks.append({"kind": "mcm", "frame": frame, "depth": depth, "instances": [patch]})
                circuits.append(qc)
    out = {"backend": backend.name, "tasks": tasks, "circuits": circuits, "shots": shots, "delta": delta,
           "depths": list(depths)}
    out["estimate"] = backend.estimate(out)
    return out


def submit(xplan, backend, manifest_dir="data/manifests/xframe", job_size=None):
    """Send the circuits (``SamplerV2``, no dynamical decoupling: IBM refuses it with feed-forward) and write the
    manifest after every job, so an interrupted session keeps its job ids."""
    import json
    from datetime import datetime
    from pathlib import Path

    from qiskit_ibm_runtime import SamplerV2

    job_size = job_size or backend.max_circuits_per_job
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(manifest_dir) / f"{stamp}_xframe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"backend": backend.name, "created": datetime.now().isoformat(), "experiment": "xframe",
                "shots": xplan["shots"], "delta": xplan["delta"], "depths": xplan["depths"], "jobs": []}
    tasks, circuits = xplan["tasks"], xplan["circuits"]
    for start in range(0, len(circuits), job_size):
        chunk = list(range(start, min(start + job_size, len(circuits))))
        job = SamplerV2(mode=backend.backend).run([circuits[i] for i in chunk], shots=xplan["shots"])
        manifest["jobs"].append({"job_id": job.job_id(), "tasks": [
            {"index": k, "frame": tasks[i]["frame"], "depth": tasks[i]["depth"],
             "instance": tasks[i]["instances"][0].to_dict()} for k, i in enumerate(chunk)]})
        print(f"  job {len(manifest['jobs'])}: {len(chunk)} circuits -> {job.job_id()}")
        path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {path}")
    return str(path)


def harvest(manifest_path, backend, data_dir="data/results"):
    """Every finished job of an X-frame manifest into ``<backend>/surface_code/xframe/<stamp>_..._xframe.json``."""
    import json
    from datetime import datetime
    from pathlib import Path

    from .analysis import make_record, save_run
    from .backends.base import reverse_keys
    from .primitives import from_dict

    manifest = json.loads(Path(manifest_path).read_text())
    service = backend.service
    records, pending = [], 0
    for job in manifest["jobs"]:
        remote = service.job(job["job_id"])
        if str(remote.status()) not in ("DONE", "JobStatus.DONE"):
            print(f"  {job['job_id']}: {remote.status()}")
            pending += 1
            continue
        result = remote.result()
        for task in job["tasks"]:
            patch = from_dict(task["instance"])
            counts = reverse_keys(result[task["index"]].data.d0.get_counts())   # character i = data qubit i
            records.append(make_record(counts, patch, depth=task["depth"], delta=manifest["delta"],
                                       backend_name=manifest["backend"], job_id=job["job_id"],
                                       kind=KINDS[task["frame"]],
                                       extra={"frame": task["frame"], "manifest": Path(manifest_path).name}))
    stamp = datetime.fromisoformat(manifest["created"]).strftime("%Y%m%d_%H%M%S")
    path = Path(data_dir) / manifest["backend"] / "surface_code" / "xframe" / \
        f"{stamp}_{manifest['backend']}_surface_code_xframe.json"
    saved = save_run(path, records, run={k: manifest[k] for k in ("backend", "created", "shots", "delta", "depths")}
                     | {"manifest": Path(manifest_path).name, "experiment": "xframe"})
    print(f"{len(records)} results written to {saved}" + (f"; {pending} job(s) not finished" if pending else ""))
    return saved


def load(path):
    """``{frame: {data-qubit tuple: {depth: r_ovl}}}`` from a harvested X-frame run file."""
    import json
    from pathlib import Path

    doc = json.loads(Path(path).read_text())
    out = {}
    for rec in doc["results"]:
        frame = rec["benchmark"].get("frame") or ("X" if rec["parameters"]["kind"] == KINDS["X"] else "Z")
        key = tuple(rec["parameters"]["data_qubits"])
        out.setdefault(frame, {}).setdefault(key, {})[rec["parameters"]["depth"]] = rec["benchmark"]["r_ovl"]
    return out, doc["run"]
