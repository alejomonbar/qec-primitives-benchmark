"""Surface-code memory experiments on the same qubits as the LR-QAOA benchmark.

The LR-QAOA benchmark reads a code as a classical Ising Hamiltonian: every check is a ``Z...Z`` term. A
**memory experiment** runs the code for real. It prepares a logical state, repeats ``R`` rounds of syndrome
extraction with the true ``X``/``Z`` checks, reads the data qubits, and asks a decoder whether the logical
state survived. Its answer, the **logical error rate** after ``R`` rounds, is what error correction is ultimately
judged by, so it is the natural yardstick for the benchmark: run both on the same patch of a chip and compare.

The code
--------
``codes.surface_code(d)`` gives the check supports on a ``d x d`` grid (data qubit ``n`` at row ``n // d``,
column ``n % d``); ``check_types`` adds the rotated-surface-code checkerboard:

* a bulk plaquette is ``X`` if the row + column of its top-left corner is even, else ``Z``;
* a weight-2 boundary check is ``Z`` if its two data qubits share a column, else ``X``.

Logical ``Z`` is row 0 of the grid, logical ``X`` column 0 (``logical_support``). ``verify_code`` checks that
every ``X``/``Z`` pair commutes and the logicals anticommute.

One round
---------
``X`` ancillas get ``H``; four CNOT layers follow, in which each ancilla touches one data qubit per layer
(``cx_schedule``: data -> ancilla for ``Z`` checks, ancilla -> data for ``X`` checks), then ``H`` on the ``X``
ancillas, a mid-circuit measurement of every ancilla and a reset. The order of the four layers is not free: the
``Z`` checks use ``Z_ORDER`` and the ``X`` checks its transpose ``X_ORDER`` (read off stim's rotated-memory
circuit); any other choice makes the detectors non-deterministic or lowers the circuit distance below ``d``.

Detectors and decoding
----------------------
``basis="Z"`` stores ``|0>_L`` and is limited by bit flips, which the ``Z`` checks catch; ``basis="X"`` stores
``|+>_L`` (``H`` on the data at both ends) and is limited by phase flips. The **detectors** are

* round 0: the checks of the stored basis (the only ones with a deterministic outcome after the reset);
* round ``r > 0``: every check, XOR the same check in round ``r - 1``;
* final: the stored-basis checks rebuilt from the data readout, XOR their last measurement.

A detector that fires marks an error nearby. ``stim_circuit`` writes the same experiment with circuit-level
noise of strength ``p_2q``; its detector error model gives the matching graph, and ``pymatching`` predicts
whether the logical operator (the parity of ``logical_support`` in the data readout) was flipped. ``p_2q``
only sets the matching weights, not the measured answer. The logical error rate is the fraction of shots the
decoder gets wrong, with shot noise ``sqrt(P (1 - P) / S)``; ``error_per_round`` turns ``P(R)`` into an error per
round ``eps`` from ``P(R) = (1 - A (1 - 2 eps)^R) / 2``, ``A`` absorbing state preparation and readout.

Circuits and data
-----------------
``memory_circuit`` builds the round on compact qubits (data ``0..n-1``, then one ancilla per check in the
order of ``code.checks``) with one classical register per round, ``s0 .. s{R-1}`` (bit ``k`` = check ``k``), and
``c`` for the data (bit ``i`` = data qubit ``i``). On a chip it is laid onto a ``CodePatch`` placement
(``layout.surface_code_placements``), so the memory and the LR-QAOA benchmark use the very same data qubits,
ancillas and couplers. A result stores every raw shot, the registers ``s0 .. s{R-1} c`` concatenated with each
written bit 0 first, packed eight bits to a byte (``pack_shots``), so it can be decoded again with any decoder.
(As text, counts of shot strings, the 228 circuits of one position scan at 4000 shots take 85 MB; packed, 13.)

The experiment itself - plan, submit, harvest, load - is ``plan``, ``submit``, ``harvest`` and ``load_results``.
Circuits contain only Clifford gates, resets and measurements, so a local run uses Aer's stabilizer simulator
and is fast at any distance, under the uniform model of ``SimBackend`` or a Pauli model built from a device's
calibration (``calibrated_noise_model``).
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

Z_ORDER = [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]
X_ORDER = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
BASES = ("Z", "X")


# ======================================================================================
# The code
# ======================================================================================
def _grid(code):
    if code.family != "surface_code":
        raise ValueError(f"{code.name}: memory experiments are implemented for surface_code structures")
    d = int(code.info.get("distance") or round(code.n_data ** 0.5))
    return d, {n: (n // d, n % d) for n in range(code.n_data)}


def check_types(code) -> tuple[str, ...]:
    """``"X"`` or ``"Z"`` for each check of ``code.checks``: the rotated-surface-code checkerboard."""
    _, rc = _grid(code)
    out = []
    for check in code.checks:
        cells = [rc[n] for n in check]
        rows = {r for r, _ in cells}
        if len(check) == 4:
            out.append("X" if (min(rows) + min(c for _, c in cells)) % 2 == 0 else "Z")
        else:
            out.append("Z" if len(rows) == 2 else "X")
    return tuple(out)


def logical_support(code, basis: str = "Z") -> list[int]:
    """Data qubits of the logical operator the memory preserves: row 0 (``Z``) or column 0 (``X``)."""
    d, _ = _grid(code)
    return list(range(d)) if basis == "Z" else [i * d for i in range(d)]


def verify_code(code):
    """Assert the typed checks form a commuting code with anticommuting logicals; returns ``(n_X, n_Z)``."""
    types = check_types(code)
    supports = [set(c) for c in code.checks]
    for i, a in enumerate(supports):
        for j in range(i + 1, len(supports)):
            if types[i] != types[j] and len(a & supports[j]) % 2:
                raise AssertionError(f"{code.name}: checks {code.checks[i]} and {code.checks[j]} anticommute")
    lz, lx = set(logical_support(code, "Z")), set(logical_support(code, "X"))
    if any(len(lz & s) % 2 for s, t in zip(supports, types) if t == "X"):
        raise AssertionError(f"{code.name}: logical Z anticommutes with an X check")
    if any(len(lx & s) % 2 for s, t in zip(supports, types) if t == "Z"):
        raise AssertionError(f"{code.name}: logical X anticommutes with a Z check")
    if len(lx & lz) % 2 != 1:
        raise AssertionError(f"{code.name}: logical X and Z commute")
    return types.count("X"), types.count("Z")


def cx_schedule(code) -> tuple[list, ...]:
    """For each check, the data qubit (or None) it touches in each of the four CNOT layers."""
    _, rc = _grid(code)
    types = check_types(code)
    out = []
    for check, kind in zip(code.checks, types):
        cells = {n: rc[n] for n in check}
        cr = np.mean([p[0] for p in cells.values()])
        cc = np.mean([p[1] for p in cells.values()])
        if len(check) == 2:              # half a plaquette: centre it on the plaquette it would complete
            if len({p[0] for p in cells.values()}) == 1:
                cr += -0.5 if cr == 0 else +0.5
            else:
                cc += -0.5 if cc == 0 else +0.5
        slot = {(int(np.sign(r - cr)), int(np.sign(c - cc))): n for n, (r, c) in cells.items()}
        out.append([slot.get(k) for k in (Z_ORDER if kind == "Z" else X_ORDER)])
    return tuple(out)


# ======================================================================================
# Reference circuit (stim): distance and decoder
# ======================================================================================
def stim_circuit(code, rounds: int, basis: str = "Z", p_2q: float = 1e-3, p_1q=None, p_spam=None, p_meas=None):
    """The memory experiment with circuit-level noise, detectors in the order of ``detection_events``."""
    import stim

    p_1q = p_2q / 10 if p_1q is None else p_1q
    p_spam = p_2q if p_spam is None else p_spam
    p_meas = p_2q if p_meas is None else p_meas
    types, order = check_types(code), cx_schedule(code)
    n, m = code.n_data, code.n_checks
    anc = [n + k for k in range(m)]
    x_anc = [anc[k] for k in range(m) if types[k] == "X"]
    c = stim.Circuit()
    c.append("R", range(n + m))
    c.append("X_ERROR", range(n + m), p_spam)
    if basis == "X":
        c.append("H", range(n))
        c.append("DEPOLARIZE1", range(n), p_1q)
    for r in range(rounds):
        c.append("H", x_anc)
        c.append("DEPOLARIZE1", x_anc, p_1q)
        for step in range(4):
            pairs = []
            for k in range(m):
                q = order[k][step]
                if q is not None:
                    pairs += [anc[k], q] if types[k] == "X" else [q, anc[k]]
            if pairs:
                c.append("CX", pairs)
                c.append("DEPOLARIZE2", pairs, p_2q)
        c.append("H", x_anc)
        c.append("DEPOLARIZE1", x_anc, p_1q)
        c.append("X_ERROR", anc, p_meas)
        c.append("MR", anc)
        c.append("X_ERROR", anc, p_spam)
        for k in range(m):
            if r == 0:
                if types[k] == basis:
                    c.append("DETECTOR", [stim.target_rec(k - m)])
            else:
                c.append("DETECTOR", [stim.target_rec(k - m), stim.target_rec(k - 2 * m)])
    if basis == "X":
        c.append("DEPOLARIZE1", range(n), p_1q)
        c.append("H", range(n))
    c.append("X_ERROR", range(n), p_spam)
    c.append("M", range(n))
    for k, check in enumerate(code.checks):
        if types[k] == basis:
            c.append("DETECTOR", [stim.target_rec(q - n) for q in check] + [stim.target_rec(k - n - m)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(q - n) for q in logical_support(code, basis)], 0)
    return c


def code_distance(code, rounds: int = 1, basis: str = "Z") -> int:
    """Graphlike distance stim finds for the memory circuit; ``d`` for a correct schedule."""
    dem = stim_circuit(code, rounds, basis).detector_error_model(decompose_errors=True)
    return len(dem.shortest_graphlike_error())


def decoder(code, rounds: int, basis: str = "Z", p_2q: float = 3e-3):
    """``pymatching`` decoder from the detector error model of ``stim_circuit`` (cached per code, rounds, basis)."""
    return _decoder(code, int(rounds), basis, float(p_2q))


@lru_cache(maxsize=64)
def _decoder(code, rounds, basis, p_2q):
    import pymatching

    dem = stim_circuit(code, rounds, basis, p_2q).detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


# ======================================================================================
# Device circuit, shots, detectors
# ======================================================================================
def memory_circuit(code, rounds: int, basis: str = "Z"):
    """The experiment on compact qubits: data ``0..n-1``, the ancilla of check ``k`` at ``n + k``.

    Registers ``s0 .. s{R-1}`` hold the syndromes of each round (bit ``k`` = check ``k``), ``c`` the data.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    types, order = check_types(code), cx_schedule(code)
    n, m = code.n_data, code.n_checks
    anc = [n + k for k in range(m)]
    x_anc = [anc[k] for k in range(m) if types[k] == "X"]
    syndromes = [ClassicalRegister(m, f"s{r}") for r in range(rounds)]
    data = ClassicalRegister(n, "c")
    qc = QuantumCircuit(QuantumRegister(n + m, "q"), *syndromes, data)
    if basis == "X":
        qc.h(range(n))
    for r in range(rounds):
        if x_anc:
            qc.h(x_anc)
        qc.barrier(anc)                  # the Z ancillas wait for the H, keeping the round at 4 CX layers
        for step in range(4):
            for k in range(m):
                q = order[k][step]
                if q is not None:
                    if types[k] == "X":
                        qc.cx(anc[k], q)
                    else:
                        qc.cx(q, anc[k])
        if x_anc:
            qc.h(x_anc)
        qc.barrier(anc)
        qc.measure(anc, syndromes[r])
        qc.reset(anc)
    if basis == "X":
        qc.h(range(n))
    qc.barrier(range(n))                 # one readout instant for all data qubits
    qc.measure(range(n), data)
    return qc


def shot_key(syndromes, data) -> str:
    """The stored form of one shot: registers ``s0 .. s{R-1} c``, each written bit 0 first."""
    return " ".join(["".join(map(str, s)) for s in syndromes] + ["".join(map(str, data))])


def shots_from_counts(counts, rounds: int, n_checks: int, n_data: int):
    """``(syndromes (S, R, m), data (S, n))`` as uint8 arrays from counts of ``shot_key`` strings."""
    keys = list(counts)
    reps = np.array([counts[k] for k in keys])
    rows = np.array([np.frombuffer(k.replace(" ", "").encode(), dtype=np.uint8) - 48 for k in keys], dtype=np.uint8)
    if rows.shape[1] != rounds * n_checks + n_data:
        raise ValueError(f"shots have {rows.shape[1]} bits, expected {rounds} x {n_checks} + {n_data}")
    rows = np.repeat(rows, reps, axis=0)
    return rows[:, :rounds * n_checks].reshape(-1, rounds, n_checks), rows[:, rounds * n_checks:]


def detection_events(syndromes, data, code, basis: str = "Z"):
    """Detector outcomes ``(S, n_detectors)`` in the order of ``stim_circuit``, and the observable ``(S,)``."""
    syndromes, data = np.asarray(syndromes, dtype=np.uint8), np.asarray(data, dtype=np.uint8)
    stored = np.array([t == basis for t in check_types(code)])
    parts = [syndromes[:, 0, stored]]
    if syndromes.shape[1] > 1:
        parts.append((syndromes[:, 1:, :] ^ syndromes[:, :-1, :]).reshape(len(data), -1))
    rebuilt = np.stack([np.bitwise_xor.reduce(data[:, list(c)], axis=1) for c in code.checks], axis=1)
    parts.append(rebuilt[:, stored] ^ syndromes[:, -1, stored])
    observable = np.bitwise_xor.reduce(data[:, logical_support(code, basis)], axis=1)
    return np.concatenate(parts, axis=1), observable


def pack_shots(syndromes, data) -> dict:
    """Every shot as its bits (syndromes round by round, then data), packed and base64-encoded."""
    import base64

    syndromes, data = np.asarray(syndromes, dtype=np.uint8), np.asarray(data, dtype=np.uint8)
    bits = np.concatenate([syndromes.reshape(len(data), -1), data], axis=1)
    return {"encoding": "numpy.packbits, base64", "shots": int(bits.shape[0]), "bits_per_shot": int(bits.shape[1]),
            "data": base64.b64encode(np.packbits(bits, axis=1).tobytes()).decode()}


def unpack_shots(packed, rounds: int, n_checks: int, n_data: int):
    """Inverse of ``pack_shots``: ``(syndromes (S, R, m), data (S, n))``."""
    import base64

    nbits = packed["bits_per_shot"]
    if nbits != rounds * n_checks + n_data:
        raise ValueError(f"shots have {nbits} bits, expected {rounds} x {n_checks} + {n_data}")
    raw = np.frombuffer(base64.b64decode(packed["data"]), dtype=np.uint8).reshape(packed["shots"], -1)
    bits = np.unpackbits(raw, axis=1)[:, :nbits]
    return bits[:, :rounds * n_checks].reshape(-1, rounds, n_checks), bits[:, rounds * n_checks:]


def record_shots(record, code):
    """``(syndromes, data)`` of a stored memory result, packed or as counts."""
    params = record["parameters"]
    if "shots_packed" in record:
        return unpack_shots(record["shots_packed"], params["rounds"], code.n_checks, code.n_data)
    return shots_from_counts(record["samples"], params["rounds"], code.n_checks, code.n_data)


def logical_error_rate(counts, code, rounds: int, basis: str = "Z", p_2q: float = 3e-3):
    """Decode shots given as counts: ``(logical error rate, its shot noise, fraction of detectors that fired)``."""
    return decode(*shots_from_counts(counts, rounds, code.n_checks, code.n_data), code, rounds, basis, p_2q)


def decode(syndromes, data, code, rounds: int, basis: str = "Z", p_2q: float = 3e-3):
    """Decode shots given as arrays: ``(logical error rate, its shot noise, fraction of detectors that fired)``."""
    events, observable = detection_events(syndromes, data, code, basis)
    predicted = decoder(code, rounds, basis, p_2q).decode_batch(events).reshape(-1)
    rate = float(np.mean(predicted != observable))
    return rate, float(np.sqrt(max(rate * (1 - rate), 1e-12) / len(observable))), float(events.mean())


def error_per_round(rounds, rates):
    """Fit ``P(R) = (1 - A (1 - 2 eps)^R) / 2`` to logical error rates: ``(eps, eps_stderr, A)``.

    ``eps`` is the logical error per round. ``A <= 1`` absorbs what does not grow with ``R`` - state preparation
    and the final readout - so it is not charged to the rounds. With fewer than three round counts ``A`` is
    fixed to 1.
    """
    from scipy.optimize import curve_fit

    rounds, rates = np.asarray(rounds, float), np.clip(np.asarray(rates, float), 0, 0.499)
    guess = max(float(np.max(rates)) / max(float(np.max(rounds)), 1.0), 1e-5)
    if len(rounds) < 3:
        (eps,), cov = curve_fit(lambda R, e: (1 - (1 - 2 * e) ** R) / 2, rounds, rates, p0=[guess], bounds=(0, 0.5))
        return float(eps), float(np.sqrt(cov[0, 0])), 1.0
    (eps, amplitude), cov = curve_fit(lambda R, e, a: (1 - a * (1 - 2 * e) ** R) / 2, rounds, rates,
                                      p0=[guess, 1.0], bounds=([0, 0], [0.5, 1]))
    return float(eps), float(np.sqrt(cov[0, 0])), float(amplitude)


# ======================================================================================
# Noise for local runs
# ======================================================================================
def calibrated_noise_model(patch, cal, default_2q: float = 1e-2, default_1q: float = 1e-3,
                           default_readout: float = 2e-2):
    """A Pauli noise model of the patch from a device calibration, on the compact qubits of ``memory_circuit``.

    Depolarizing error on every CNOT from the calibrated two-qubit error of its coupler, on ``H`` from the
    ``sx`` error of the qubit, and readout error from the mid-circuit readout of the ancillas and the final
    readout of the data. Idling and crosstalk are left out, so it is optimistic; it is Clifford, so the
    stabilizer simulator runs it at any distance.
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    one, two = cal["one_qubit"], cal["two_qubit"]
    q = lambda x: one.get(x) or one.get(str(x)) or {}
    index = {phys: i for i, phys in enumerate(patch.qubits)}
    model = NoiseModel(basis_gates=["cx", "h", "measure", "reset", "barrier"])
    for u, v in patch.couplers:
        entry = two.get(f"{u}-{v}") or two.get(f"{v}-{u}") or {}
        p = entry.get("error") or default_2q
        for a, b in ((u, v), (v, u)):
            model.add_quantum_error(depolarizing_error(min(p, 0.75), 2), ["cx"], [index[a], index[b]])
    ancillas = set(patch.ancillas)
    for phys, i in index.items():
        p1 = q(phys).get("sx_error") or default_1q
        model.add_quantum_error(depolarizing_error(min(p1, 0.75), 1), ["h"], [i])
        ro = (q(phys).get("mcm_readout_error") if phys in ancillas else None) or q(phys).get("readout_error") or default_readout
        model.add_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [i])
    return model


def uniform_noise_model(p_1q: float, p_2q: float, readout: float):
    """The ``SimBackend`` model restricted to the gates of ``memory_circuit``."""
    from .noise import depolarizing_noise_model

    return depolarizing_noise_model(p_1q=p_1q, p_2q=p_2q, readout=readout, one_qubit_gates=("h",),
                                    two_qubit_gates=("cx",))


# ======================================================================================
# Experiment: plan -> submit (manifest) -> harvest -> load
# ======================================================================================
def _is_device(backend):
    return hasattr(backend, "backend") and not getattr(backend, "local", True)


def plan(backend, patches, rounds, bases=BASES, shots: int = 4000, p_2q_model: float = 3e-3):
    """Every (patch, basis, rounds) circuit. On a device, each is transpiled onto the patch's own qubits and
    refused if the transpiler needed any coupler outside the patch."""
    from qiskit import transpile

    tasks, circuits = [], []
    for patch in patches:
        if patch.kind != "mcm" or patch.logical:
            raise ValueError(f"{patch}: a memory experiment needs a physical mcm placement (one ancilla per check)")
        verify_code(patch.code)
        for basis in bases:
            for R in rounds:
                qc = memory_circuit(patch.code, R, basis)
                if _is_device(backend):
                    qc = transpile(qc, backend=backend.backend, initial_layout=list(patch.qubits),
                                   optimization_level=backend.optimization_level)
                    allowed = {tuple(sorted(c)) for c in patch.couplers}
                    routed = {tuple(sorted(qc.find_bit(b).index for b in inst.qubits)) for inst in qc.data
                              if len(inst.qubits) == 2 and inst.operation.name not in ("barrier",)} - allowed
                    if routed:
                        raise ValueError(f"{patch}, R = {R}: routing on {sorted(routed)[:3]}")
                tasks.append({"instance": patch, "basis": basis, "rounds": R})
                circuits.append(qc)
    out = {"backend": backend.name, "tasks": tasks, "circuits": circuits, "shots": shots, "rounds": list(rounds),
           "bases": list(bases), "p_2q_model": p_2q_model, "estimate": None}
    if _is_device(backend) and hasattr(backend, "circuit_seconds"):
        seconds = [backend.circuit_seconds(qc) * shots for qc in circuits]
        out["estimate"] = {"unit": "QPU s", "total": sum(seconds), "per_circuit": seconds,
                           "note": "gate and readout durations only; the wait between shots is not included"}
    return out


def print_plan(memory_plan):
    tasks = memory_plan["tasks"]
    patches = {t["instance"] for t in tasks}
    print(f"Backend  : {memory_plan['backend']}")
    print(f"Patches  : {len(patches)} ({', '.join(sorted({p.code.name for p in patches}))})")
    print(f"Sweep    : rounds {memory_plan['rounds']}, bases {memory_plan['bases']}, {memory_plan['shots']} shots each")
    print(f"Circuits : {len(tasks)}")
    est = memory_plan.get("estimate")
    if est:
        print(f"Estimate : {est['total']:.1f} {est['unit']} ({est['note']})")


def submit(memory_plan, backend, manifest_dir="data/manifests", noise_model=None, seed=None, job_size: int = 100):
    """Send the circuits (or simulate them locally) and write a manifest after every job.

    Locally, ``noise_model`` is an Aer noise model or a function of the patch returning one (e.g.
    ``lambda patch: calibrated_noise_model(patch, cal)``, since every patch sees its own qubits).
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(manifest_dir) / backend.name / f"{stamp}_{backend.name}_surface_code_memory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"backend": backend.name, "created": datetime.now().isoformat(), "experiment": "memory",
                "shots": memory_plan["shots"], "rounds": memory_plan["rounds"], "bases": memory_plan["bases"],
                "p_2q_model": memory_plan["p_2q_model"], "simulated": not _is_device(backend),
                "noise_model": None if _is_device(backend) else _noise_text(backend, noise_model), "jobs": []}
    circuits, tasks = memory_plan["circuits"], memory_plan["tasks"]
    for start in range(0, len(circuits), job_size):
        chunk = list(range(start, min(start + job_size, len(circuits))))
        record = {"tasks": [{"index": k, "basis": tasks[i]["basis"], "rounds": tasks[i]["rounds"],
                             "instance": tasks[i]["instance"].to_dict()} for k, i in enumerate(chunk)]}
        if _is_device(backend):
            from qiskit_ibm_runtime import SamplerV2

            job = SamplerV2(mode=backend.backend).run([circuits[i] for i in chunk], shots=memory_plan["shots"])
            record["job_id"] = job.job_id()
            print(f"  job {len(manifest['jobs']) + 1}: {len(chunk)} circuits -> {record['job_id']}")
        else:
            record["job_id"] = f"{backend.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
            record["counts"] = [_simulate(circuits[i], memory_plan["shots"],
                                          noise_model(tasks[i]["instance"]) if callable(noise_model) else noise_model,
                                          None if seed is None else seed + i) for i in chunk]
            print(f"  simulated {len(chunk)} circuits -> {record['job_id']}")
        manifest["jobs"].append(record)
        path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {path}")
    return str(path)


def _noise_text(backend, noise_model):
    if noise_model is None:
        return "Aer stabilizer simulator, noiseless"
    if callable(noise_model):
        return ("Aer stabilizer simulator, Pauli model of each patch from the device calibration "
                "(qecbench.memory.calibrated_noise_model: no idling or crosstalk)")
    return f"Aer stabilizer simulator, {getattr(backend, 'noise_description', None) or 'Pauli noise model supplied by the caller'}"


def _simulate(circuit, shots, noise_model, seed):
    from qiskit_aer import AerSimulator

    result = AerSimulator(method="stabilizer", noise_model=noise_model).run(
        circuit, shots=shots, memory=True, seed_simulator=seed).result()
    counts = {}
    for shot in result.get_memory():                   # registers in reverse order, each MSB first
        key = " ".join(reg[::-1] for reg in reversed(shot.split()))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _device_counts(pub, rounds):
    registers = [getattr(pub.data, f"s{r}").get_bitstrings() for r in range(rounds)] + [pub.data.c.get_bitstrings()]
    counts = {}
    for shot in zip(*registers):
        key = " ".join(bits[::-1] for bits in shot)
        counts[key] = counts.get(key, 0) + 1
    return counts


def harvest(manifest_path, backend, data_dir="data/results", service=None):
    """Decode every finished job of a manifest into ``<backend>/surface_code/memory/<stamp>_..._memory.json``."""
    from .analysis import save_run
    from .primitives import from_dict

    manifest = json.loads(Path(manifest_path).read_text())
    stamp = datetime.fromisoformat(manifest["created"]).strftime("%Y%m%d_%H%M%S")
    records = []
    for job in manifest["jobs"]:
        if "counts" in job:
            per_task = job["counts"]
        else:
            if service is None:
                service = getattr(backend, "service", None)
            remote = service.job(job["job_id"])
            if str(remote.status()) not in ("DONE", "JobStatus.DONE"):
                print(f"  {job['job_id']}: {remote.status()}")
                continue
            result = remote.result()
            per_task = [_device_counts(result[t["index"]], t["rounds"]) for t in job["tasks"]]
        for task, counts in zip(job["tasks"], per_task):
            patch = from_dict(task["instance"])
            records.append(memory_record(counts, patch, task["rounds"], task["basis"], manifest, job["job_id"]))
    header = {k: manifest.get(k) for k in ("backend", "created", "experiment", "shots", "rounds", "bases",
                                           "p_2q_model", "simulated", "noise_model")}
    header["manifest"] = Path(manifest_path).name
    path = Path(data_dir) / manifest["backend"] / "surface_code" / "memory" / \
        f"{stamp}_{manifest['backend']}_surface_code_memory.json"
    saved = save_run(path, records, run=header)
    print(f"{len(records)} memory results written to {saved}")
    return saved


def memory_record(counts, patch, rounds, basis, manifest, job_id, extra=None):
    """One stored memory result: raw shots (packed), the decoded logical error rate and where it came from."""
    from .analysis import public_job_id

    syndromes, data = shots_from_counts(counts, rounds, patch.code.n_checks, patch.code.n_data)
    rate, err, fired = decode(syndromes, data, patch.code, rounds, basis, manifest["p_2q_model"])
    return {
        "metadata": {"timestamp": datetime.now().isoformat(), "experiment_type": "surface_code_memory",
                     "backend": manifest["backend"], "task_id": public_job_id(job_id), "package": "qecbench",
                     "simulated": manifest["simulated"], "noise_model": manifest["noise_model"]},
        "parameters": {"kind": f"memory_{basis.lower()}", "basis": basis, "rounds": rounds, "depth": rounds,
                       "shots": int(sum(counts.values())), "p_2q_model": manifest["p_2q_model"],
                       "data_qubits": list(patch.data_qubits), "ancillary_qubits": list(patch.ancillas),
                       "registers": "s0 .. s{R-1} c, each written bit 0 first (bit k of s_r = check k)"},
        "shots_packed": pack_shots(syndromes, data),
        "benchmark": {"family": "code", "instance": patch.to_dict(), "qubits": list(patch.qubits),
                      "logical_error_rate": rate, "logical_error_rate_err": err, "detection_fraction": fired,
                      "decoder": "pymatching on the stim detector error model (p_2q_model sets the weights)",
                      **(extra or {})},
    }


def load_results(data_dir, backend_name, files=None, p_2q_model=None, bases=None):
    """``{patch: {basis: {rounds: summary}}}``, decoded again from the stored shots.

    ``summary`` holds ``rate, err, detection_fraction, shots, file``. ``p_2q_model`` overrides the matching
    weights the run was decoded with; ``bases`` keeps only those bases.
    """
    from .analysis import instance_from_record, records_in

    out = {}
    folder = Path(data_dir) / backend_name / "surface_code" / "memory"
    for path in sorted(folder.glob("*.json")):
        if files is not None and path.name not in files:
            continue
        for record in records_in(json.loads(path.read_text())):
            params = record["parameters"]
            if bases is not None and params["basis"] not in bases:
                continue
            patch = instance_from_record(record)
            p = p_2q_model or params["p_2q_model"]
            rate, err, fired = decode(*record_shots(record, patch.code), patch.code, params["rounds"], params["basis"], p)
            out.setdefault(patch, {}).setdefault(params["basis"], {})[params["rounds"]] = {
                "rate": rate, "err": err, "detection_fraction": fired, "shots": params["shots"], "file": path.name}
    return out
