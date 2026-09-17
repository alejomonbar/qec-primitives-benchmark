"""From counts to approximation ratios, result files, and back.

Result files keep the schema of ``utils.save_experiment_results`` (``metadata``,
``parameters``, ``hamiltonian``, ``samples``, ``energy_analysis``, ``random_baseline``,
``statistical_comparison``) so the existing figure notebooks keep working, plus a
``benchmark`` block with what this package adds: the exact noiseless reference ``r_ideal``,
the normalised ``r_ovl = (r - r_rand) / (r_ideal - r_rand)`` and shot-noise error bars.

``load_results`` reads both new files (``*_chain_<qubits>_<kind>_nq<n>_depth<p>.json``) and
the legacy triplet files (``*_tri_<d1>_<a>_<d2>_<kind>_nq2_depth<p>.json``) of the MCM repo,
recomputing every derived number from the stored samples so old and new runs are compared
on exactly the same footing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .lrqaoa import angles, bitstring_energies, energy, ideal_r, random_baseline
from .primitives import FAMILIES

FILE_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4,6})_(?P<backend>.+?)_(?P<family>tri|chain|direct)_"
                     r"(?P<qubits>\d+(?:_\d+)*)_(?P<kind>[a-z_]+?)_nq(?P<n>\d+)_depth(?P<depth>\d+)\.json$")


def analyse(counts, primitive, depth, delta=0.5) -> dict:
    """Approximation ratio with its shot-noise error and both reference points."""
    ham = primitive.hamiltonian
    shots = sum(counts.values())
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    e = dict(zip(counts, bitstring_energies(counts, ham).tolist()))
    values = np.repeat(list(e.values()), list(counts.values()))
    mean_e = float(values.mean())
    err_e = float(values.std(ddof=1) / np.sqrt(shots)) if shots > 1 else float("nan")
    span = e_opt - e_max
    r = (mean_e - e_max) / span
    r_rand, r_rand_std = random_baseline(primitive, shots)
    r_id = ideal_r(primitive, depth, delta)
    return {"shots": shots, "expected_energy": mean_e, "optimal_energy": e_opt, "max_energy": e_max,
            "r": r, "r_err": abs(err_e / span), "r_rand": r_rand, "r_rand_std": r_rand_std,
            "r_ideal": r_id, "r_ovl": (r - r_rand) / (r_id - r_rand),
            "n_sigmas": (r - r_rand) / r_rand_std,
            "p_optimal": sum(n for b, n in counts.items() if abs(e[b] - e_opt) < 1e-9) / shots,
            "energies": e}


def result_dir(data_dir, backend_name, primitive, kind):
    """``<data_dir>/<backend>/<structure>/<kind>`` - where a result file belongs.

    Splitting by device, by problem and by implementation keeps a campaign navigable: one
    device's chain results are one directory, and a surface-code run later lands beside them
    instead of in the same heap.  File names stay self-describing, so a file still identifies
    itself if it is moved or copied elsewhere.
    """
    return Path(data_dir) / backend_name / primitive.structure / kind


def result_filename(backend_name, primitive, kind, depth, stamp=None):
    """Name of a single-result file (the layout earlier campaigns used)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return (f"{stamp}_{backend_name}_{primitive.family}_{primitive.tag}_{kind}"
            f"_nq{primitive.n_data}_depth{depth}.json")


def run_filename(backend_name, structure, kind, stamp=None):
    """Name of a run file: every result of one submission, for one structure and kind."""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{backend_name}_{structure}_{kind}.json"


def record_key(record):
    """What makes a result unique: the job it came from, the kind, the depth and the qubits."""
    meta, params = record.get("metadata", {}), record["parameters"]
    instance = instance_from_record(record)
    return (meta.get("task_id"), params.get("kind"), params.get("depth"),
            instance.identity if instance else tuple(params.get("data_qubits", [])))


def records_in(document):
    """The results held by a file, whether it is a run file or a single result."""
    if isinstance(document, dict) and isinstance(document.get("results"), list):
        return [r for r in document["results"] if isinstance(r, dict) and "parameters" in r]
    return [document] if isinstance(document, dict) and "parameters" in document else []


def instance_from_record(record):
    """Rebuild the primitive a result belongs to, from the file alone (None if it cannot be).

    Returns None rather than raising for records this package cannot describe - an older
    campaign storing a long chain as two unordered qubit lists, say, or some unrelated JSON
    that happens to sit in the same directory.  Callers skip those.
    """
    benchmark = record.get("benchmark") or {}
    params = record.get("parameters", {})
    data, ancillas = params.get("data_qubits") or [], params.get("ancillary_qubits") or []
    try:
        if benchmark.get("family") == "code" and benchmark.get("instance"):
            from .primitives import from_dict

            return from_dict(benchmark["instance"])
        if benchmark.get("family") in FAMILIES and benchmark.get("qubits"):
            return FAMILIES[benchmark["family"]](benchmark["qubits"])
        if ancillas and len(ancillas) == len(data) - 1:   # files written before "benchmark"
            return FAMILIES["chain"].from_roles(data, ancillas)
        return FAMILIES["direct"](data) if len(data) > 1 and not ancillas else None
    except (TypeError, ValueError):
        return None


def save_run(path, records, run=None):
    """Write or extend the run file at ``path``; results already in it are left alone.

    Harvesting the same manifest twice, or harvesting it again once more jobs have finished,
    therefore adds only what is new instead of writing another pile of files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"run": {}, "results": []}
    if path.exists():
        try:
            document = json.loads(path.read_text())
        except ValueError:
            pass
        document.setdefault("results", [])
        document.setdefault("run", {})
    if run:
        document["run"] = {**document["run"], **run}
    seen = {record_key(r) for r in records_in(document)}
    for record in records:
        if record_key(record) not in seen:
            document["results"].append(record)
            seen.add(record_key(record))
    document["results"].sort(key=lambda r: (r["parameters"]["depth"],
                                            tuple(r.get("benchmark", {}).get("qubits", ()))))
    document["run"]["results"] = len(document["results"])
    path.write_text(json.dumps(document, indent=1, default=_json_default))
    return str(path)


def make_record(counts, primitive, *, depth, delta, backend_name, job_id, kind, batch=None,
                batch_size=None, simulated=False, noise=None, extra=None) -> dict:
    """One analysed result file.

    ``simulated`` marks results that came from a simulator rather than the device, and
    ``noise`` says where that simulation's noise came from; both are written into the file so
    a run can never be mistaken for hardware data later.
    """
    a = analyse(counts, primitive, depth, delta)
    gammas, betas = angles(depth, delta)
    threshold = a["r_rand"] + 3 * a["r_rand_std"]
    return {
        "metadata": {"timestamp": datetime.now().isoformat(),
                     "experiment_type": "Hamiltonian_QAOA_mid_circuit_measurement",
                     "problem": "Hamiltonian QAOA", "backend": backend_name, "task_id": job_id,
                     "package": "qecbench", "simulated": bool(simulated),
                     "noise_model": noise},
        "parameters": {"depth": depth, "delta": delta, "num_data_qubits": primitive.n_data,
                       "data_qubits": list(primitive.data_qubits),
                       "ancillary_qubits": list(primitive.ancillas), "gammas": gammas,
                       "betas": betas, "circuit_index": batch, "shots": a["shots"], "kind": kind},
        "hamiltonian": {"num_nodes": primitive.n_data,
                        "hamiltonian_couplings": [list(t) for t in primitive.hamiltonian],
                        "hamiltonian_weights": list(primitive.hamiltonian.values())},
        "samples": dict(counts),
        "energy_analysis": {"energy_values": a["energies"], "optimal_energy": a["optimal_energy"],
                            "expected_energy": round(a["expected_energy"], 6),
                            "approximation_ratio": round(a["r"], 6),
                            "optimal_probability": round(a["p_optimal"], 6)},
        "random_baseline": {"method": "exact uniform distribution",
                            "shots_per_bootstrap": a["shots"],
                            "r_mean_of_means": round(a["r_rand"], 6),
                            "r_std_of_means": round(a["r_rand_std"], 6),
                            "r_threshold_3sigma": round(threshold, 6)},
        "statistical_comparison": {"passes_3sigma_test": bool(a["r"] > threshold),
                                   "n_sigmas_above_random": round(a["n_sigmas"], 4),
                                   "qaoa_better_than_random": bool(a["r"] > a["r_rand"])},
        "benchmark": {"family": primitive.family, "qubits": list(primitive.qubits), "kind": kind,
                      **({"instance": primitive.to_dict()} if primitive.family == "code" else {}),
                      "r": a["r"], "r_err": a["r_err"], "r_ideal": a["r_ideal"],
                      "r_ovl": a["r_ovl"], "instances_in_circuit": batch_size,
                      "simulated": bool(simulated), **(extra or {})},
    }


LEGACY_CHAIN_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4,6})_(?P<backend>.+?)_(?:(?P<token>1d[a-z]*)_)?"
                             r"(?P<variant>normal|MCM)_nq(?P<n>\d+)_depth(?P<depth>\d+)\.json$")
LEGACY_LAYOUT_TOKENS = {"1d": "free", "1dfree": "free", "1dpin": "pinned", "1dfix": "frozen", None: "unstated"}
LEGACY_KINDS = {"normal": "direct", "MCM": "mcm"}
LEGACY_EMULATORS = {                     # hosted emulators: their results are simulations, flagged as such
    "Helios-1E": "Helios-1E: Quantinuum's hosted emulator and its Helios noise model",
    "H2-1E": "H2-1E: Quantinuum's hosted emulator and its H2 noise model",
}
LEGACY_PAIR_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4,6})_(?P<backend>[^_]+)_nq2_depth(?P<depth>\d+)\.json$")
LEGACY_TRIPLET_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4,6})_(?P<backend>.+?)_tri_(?P<d1>\d+)_(?P<a>\d+)_(?P<d2>\d+)_"
                               r"(?P<variant>mcm)_nq2_depth(?P<depth>\d+)\.json$")


def convert_legacy_chain(record, filename):
    """A 1D-chain result file of the MCM repository, in this package's form. Two namings are read:

    * ``<stamp>_<backend>_[1d|1dpin|1dfix|1dfree_]<normal|MCM>_nq<n>_depth<p>.json`` - chains;
    * ``<stamp>_<backend>_tri_<d1>_<a>_<d2>_mcm_nq2_depth<p>.json`` - triplets (the ``n = 2`` chain
      ``d1 - a - d2``) from the triplet campaigns. The qubits in the name must match the record;
    * ``<stamp>_<backend>_nq2_depth<p>.json`` - the early two-spin runs (Helios-1E), stored in the MaxCut
      schema (``graph``, ``maxcut_analysis``). For one edge the cut ratio and the Ising ``r`` are the
      same number, ``P(01) + P(10)``, and both are recomputed from the samples anyway. The kind
      follows from the record: one ancilla is ``mcm``, none is ``direct``.

    Unlike moving a file, this reinterprets it:

    * ``normal`` is the ``direct`` kind and ``MCM`` the ``mcm`` kind, so both land in
      ``<backend>/chain/<kind>/``;
    * everything derived (``r``, its shot-noise error, the exact noiseless ``r_ideal``, the exact
      random baseline and ``r_ovl``) is recomputed from the stored samples, replacing the
      bootstrap baseline and simulator references of the original file;
    * a device that reuses ancillas (Quantinuum) stores either none or a pool smaller than one per
      bond. Such a chain is given logical ancilla labels following the data qubits, ``n .. 2n-2``,
      and ``benchmark.ancilla_labels`` says so, keeping the stored pool. The circuit and the
      result do not depend on those labels;
    * the per-bitstring ``energy_values`` table is not kept, since it follows from the samples;
    * runs on a hosted emulator (``LEGACY_EMULATORS``) are flagged ``simulated`` with their noise model.

    Returns ``(stamp, record)``, or ``None`` for a file this cannot describe: another variant such
    as ``MCM-DD`` or ``mcm_reset`` (which this package does not build), or a Hamiltonian that is not
    the unit-coupling chain on the stored data qubits (a surface-code file, say).
    """
    from .primitives import Chain, Direct

    name = Path(filename).name
    m = LEGACY_CHAIN_RE.match(name) or LEGACY_TRIPLET_RE.match(name) or LEGACY_PAIR_RE.match(name)
    if not m or "samples" not in record:
        return None
    triplet = "d1" in m.re.groupindex
    pair = m.re is LEGACY_PAIR_RE
    params, meta = record["parameters"], record.get("metadata", {})
    if pair:
        kind = "mcm" if len(params.get("ancillary_qubits") or []) == 1 else "direct"
    else:
        kind = "mcm" if triplet else LEGACY_KINDS[m["variant"]]
    data = [int(q) for q in params["data_qubits"]]
    if triplet and ([int(m["d1"]), int(m["d2"])] != data
                    or [int(m["a"])] != [int(q) for q in params.get("ancillary_qubits") or []]):
        return None
    ham = record.get("hamiltonian") or (
        {"hamiltonian_couplings": record["graph"]["edges"], "hamiltonian_weights": [1] * len(record["graph"]["edges"])}
        if "graph" in record else {})
    bonds = [sorted(map(int, b)) for b in ham.get("hamiltonian_couplings", [])]
    if (bonds != [[i, i + 1] for i in range(len(data) - 1)]
            or any(float(w) != 1.0 for w in ham.get("hamiltonian_weights", []))):
        return None
    ancillas = [int(q) for q in params.get("ancillary_qubits") or []]
    quantinuum = str(meta.get("backend") or m["backend"]).startswith(("Helios", "H1", "H2"))
    labels = ("logical (the device allocates the ancilla; no fixed physical ancilla)"
              if (pair or quantinuum) and kind == "mcm" else "physical")
    try:
        if kind == "direct":
            instance = Direct(data)
        else:
            if len(ancillas) != len(data) - 1:
                labels = ("logical (a fresh ancilla per gadget; no fixed physical ancilla)" if not ancillas else
                          f"logical (the device reused a pool of {len(ancillas)} ancillas: {ancillas})")
                ancillas = list(range(max(data) + 1, max(data) + len(data)))
            instance = Chain.from_roles(data, ancillas)
    except ValueError:
        return None
    samples = dict(record["samples"])
    if list(instance.data_qubits) == data[::-1] and data != data[::-1]:
        samples = {b[::-1]: c for b, c in samples.items()}     # canonical orientation reversed it
    elif list(instance.data_qubits) != data:
        return None
    token = None if (triplet or pair) else m["token"]
    backend = meta.get("backend") or m["backend"]
    converted = make_record(
        samples, instance, depth=int(params["depth"]), delta=float(params.get("delta") or 0.5),
        backend_name=backend, job_id=meta.get("task_id"), kind=kind,
        simulated=backend in LEGACY_EMULATORS, noise=LEGACY_EMULATORS.get(backend),
        extra={"source_file": Path(filename).name, "ancilla_labels": labels,
               "layout": "triplet selection" if triplet else LEGACY_LAYOUT_TOKENS.get(token, token),
               **({"source_schema": "maxcut"} if "maxcut_analysis" in record else {})})
    converted["energy_analysis"].pop("energy_values", None)
    for key in ("layout_source", "layout_frozen", "circuit_variant", "ancilla_reset"):
        if params.get(key) is not None:
            converted["parameters"][key] = params[key]
    converted["metadata"]["timestamp"] = meta.get("timestamp")
    converted["metadata"]["converted"] = datetime.now().isoformat(timespec="seconds")
    return m["stamp"], converted



LEGACY_CODE_RE = re.compile(r"^(?P<stamp>\d{8}_\d{4,6})_(?P<backend>.+?)_(?P<family>sc|cc|qldpc)(?:_(?P<code>[A-Z]{2}\d+))?"
                            r"(?:_(?P<v1>MCM|normal))?_nq(?P<n>\d+)(?:_(?P<v2>MCM|normal))?_depth(?P<depth>\d+)"
                            r"(?:_(?P<suffix>combined))?\.json$")
LEGACY_CODE_FAMILIES = {"sc": "surface_code", "cc": "color_code", "qldpc": "qldpc"}
LEGACY_CODE_PROGRAMS = {       # the Guppy programs of the MCM repository that produced the device runs
    ("sc", "mcm"): "qaoa_mcm_sc", ("sc", "direct"): "qaoa_normal_sc",
    ("cc", "mcm"): "qaoa_mcm_cc", ("cc", "direct"): "qaoa_normal_cc",
    ("qldpc", "mcm"): "qaoa_mcm_qldpc", ("qldpc", "direct"): "qaoa_normal_qldpc",
}


def legacy_code_structure(family_token, n_data, hamiltonian, code_name=None):
    """The generated structure (``codes``) a legacy code file ran, or None if its checks differ.

    ``family_token`` is ``sc``, ``cc`` or ``qldpc`` and ``hamiltonian`` the file's
    ``{support: weight}``. The size follows from ``n_data`` (``d^2`` data qubits for the surface code,
    ``(3d^2 + 1)/4`` for the colour code) or from ``code_name`` (``BB18``). The checks must agree as a
    set, weights included; their order does not matter.
    """
    from . import codes

    try:
        if family_token == "sc":
            structure = codes.surface_code(int(round(n_data ** 0.5)))
        elif family_token == "cc":
            structure = codes.color_code(int(round(((4 * n_data - 1) / 3) ** 0.5)))
        else:
            structure = codes.bivariate_bicycle(code_name or f"BB{n_data}")
    except (KeyError, ValueError):
        return None
    ours = {frozenset(c): w for c, w in structure.hamiltonian.items()}
    theirs = {frozenset(int(q) for q in c): float(w) for c, w in hamiltonian.items()}
    return structure if structure.n_data == n_data and ours == theirs else None


def convert_legacy_code(record, filename, placement=None):
    """A code-structure result file of the MCM repository (``benchmarking_quantinuum.ipynb``,
    ``qldpc_code.ipynb``) in this package's form, or None.

    Names read: ``<stamp>_<backend>_<sc|cc>_<MCM|normal>_nq<n>_depth<p>.json``, the early
    ``<stamp>_<backend>_sc_nq<n>[_MCM]_depth<p>.json``, and
    ``<stamp>_<backend>_qldpc_<BB..>_<MCM|normal>_nq<n>_depth<p>[_combined].json`` (``combined`` merges
    the samples of several jobs; ``metadata.merged_from`` names them). A name with no ``MCM`` or
    ``normal`` token is read as ``direct``, as the paper figures did, and ``benchmark.kind_source``
    records that it was inferred.

    As for chains, ``normal`` becomes ``direct``, everything derived is recomputed from the samples,
    ``energy_values`` is dropped and hosted-emulator runs are flagged ``simulated``. The backend is the one in
    the file name; where the metadata names another, ``benchmark.backend_note`` says so. The instance is a
    logical ``CodePatch`` of the generated structure (``legacy_code_structure``), whose checks must
    equal the file's Hamiltonian. Simulator files (the noiseless references of that repository) are
    not device runs and give None.

    The Quantinuum programs measured the checks one at a time, a fresh ancilla each (``benchmark.program``
    names the function), unlike the batched programs of ``code_programs``.

    A run on a fixed-layout chip (IBM) stores logical labels only. ``placement`` - the physical ``CodePatch``
    it ran on, e.g. from ``layout.surface_code_placements(G, code, anchors=[(r0, c0)])`` - puts it back on its
    qubits; without one such a file gives None. A name tagged after the device (``ibm_phoenix-scanbest``) is
    filed under the device of its metadata, the tag kept as ``benchmark.run_tag``.
    """
    from .primitives import CodePatch

    name = Path(filename).name
    m = LEGACY_CODE_RE.match(name)
    if not m or "samples" not in record or "hamiltonian" not in record:
        return None
    meta, params = record.get("metadata", {}), record["parameters"]
    backend = m["backend"]                      # the name wins: one Helios-1E file says Helios-1 inside
    if "simulator" in backend or "simulator" in str(meta.get("backend")):
        return None
    run_tag = None
    if meta.get("backend") and backend.startswith(f"{meta['backend']}-"):
        backend, run_tag = meta["backend"], backend[len(meta["backend"]) + 1:]
    physical = backend.startswith("ibm_")
    token = m["v1"] or m["v2"]
    kind = LEGACY_KINDS[token] if token else "direct"
    n_data = int(params["num_data_qubits"])
    if [int(q) for q in params["data_qubits"]] != list(range(n_data)) or n_data != int(m["n"]):
        return None
    ham = record["hamiltonian"]
    hamiltonian = dict(zip((tuple(c) for c in ham["hamiltonian_couplings"]),
                           ham.get("hamiltonian_weights") or [1] * len(ham["hamiltonian_couplings"])))
    structure = legacy_code_structure(m["family"], n_data, hamiltonian, m["code"])
    if structure is None:
        return None
    if physical:
        if placement is None or placement.kind != kind or placement.code != structure:
            return None
        instance = placement
        extra = {"source_file": name, "program": "nighthawk_sc.sc_lrqaoa",
                 "circuit_variant": "square-lattice embedding, the CZ of every check in 4 rounds",
                 "placement": "physical qubits rebuilt from the patch anchor; the file stores logical labels"}
    else:
        instance = CodePatch(structure, kind=kind)
        extra = {"source_file": name, "program": LEGACY_CODE_PROGRAMS[(m["family"], kind)],
                 "circuit_variant": "serial (one check at a time" + (", a fresh ancilla each)" if kind == "mcm" else ")"),
                 **({"ancilla_labels": "logical (a fresh ancilla per check; no fixed physical ancilla)"}
                    if kind == "mcm" else {})}
    extra["kind_source"] = "file name" if token else "inferred: no MCM or normal token in the file name"
    if run_tag:
        extra["run_tag"] = run_tag
    if meta.get("merged_from"):
        extra["merged_from"] = list(meta["merged_from"])
    if meta.get("backend") and meta["backend"] != backend and not run_tag:
        extra["backend_note"] = f"metadata says {meta['backend']}, the file name {backend}; the name is used"
    converted = make_record(
        dict(record["samples"]), instance, depth=int(params["depth"]), delta=float(params.get("delta") or 0.5),
        backend_name=backend, job_id=meta.get("task_id"), kind=kind,
        simulated=backend in LEGACY_EMULATORS, noise=LEGACY_EMULATORS.get(backend), extra=extra)
    converted["energy_analysis"].pop("energy_values", None)
    converted["metadata"]["timestamp"] = meta.get("timestamp")
    converted["metadata"]["converted"] = datetime.now().isoformat(timespec="seconds")
    return m["stamp"], converted


def save_record(record, data_dir, filename):
    path = Path(data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=_json_default))
    return str(path)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def parse_filename(name):
    m = FILE_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {"backend": d["backend"], "qubits": tuple(int(q) for q in d["qubits"].split("_")),
            "kind": d["kind"], "n_data": int(d["n"]), "depth": int(d["depth"]),
            "family": "chain" if d["family"] == "tri" else d["family"],
            "legacy": d["family"] == "tri"}


def load_results(data_dir, backend_name, kind="mcm", n_data=None, job_ids=None, manifest_path=None,
                 files=None):
    """``{Chain: {depth: summary}}`` from every result file of a backend in ``data_dir``.

    Several campaigns on one instance/depth: the newest file wins (file names start with
    the timestamp).  ``manifest_path`` (or ``job_ids``) keeps only that submission's jobs.
    ``summary`` holds ``r, r_err, r_ideal, r_ovl, r_rand, r_rand_std, shots, job_id, file``.

    Both layouts are read: the ``<backend>/<structure>/<kind>/`` tree this package writes, and a
    flat directory of result files (how earlier campaigns stored them). ``files`` keeps only the
    files with those names, to pick particular runs.
    """
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text())
        job_ids = {j["job_id"] for j in manifest["jobs"] if j.get("job_id")}
    root = Path(data_dir)
    scope = root / backend_name if (root / backend_name).is_dir() else root
    results = {}
    for path in sorted(scope.rglob("*.json"), key=lambda p: p.name):
        if files is not None and path.name not in files:
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for record in records_in(document):
            meta, params = record.get("metadata", {}), record["parameters"]
            if meta.get("backend") != backend_name:
                continue
            from_name = parse_filename(path.name) or {}
            if (params.get("kind") or from_name.get("kind") or "mcm") != kind:
                continue
            if job_ids is not None and meta.get("task_id") not in job_ids:
                continue
            instance = instance_from_record(record)
            if instance is None or (n_data is not None and instance.n_data != n_data):
                continue
            depth = params["depth"]
            a = analyse(record["samples"], instance, depth, params.get("delta") or 0.5)
            a.pop("energies")
            results.setdefault(instance, {})[depth] = {
                **a, "job_id": meta.get("task_id"), "file": path.name,
                "legacy": "benchmark" not in record}
    return results


def ancilla_scores(results, depth=None, value="r_ovl", reduce="max"):
    """Per-ancilla score for device maps: every ancilla of an instance gets its value.

    ``depth=None`` takes the best depth.  ``reduce`` combines instances sharing an ancilla.
    """
    per = {}
    for inst, by_depth in results.items():
        if depth is None:
            v = max(s[value] for s in by_depth.values())
        elif depth in by_depth:
            v = by_depth[depth][value]
        else:
            continue
        for a in inst.ancillas:
            per.setdefault(a, []).append(v)
    agg = {"max": max, "min": min, "mean": lambda v: sum(v) / len(v)}[reduce]
    return {a: agg(v) for a, v in per.items()}


def depth_summary(results, value="r"):
    """``{depth: (median, q25, q75, n)}`` over instances."""
    depths = sorted({p for by in results.values() for p in by})
    out = {}
    for p in depths:
        v = np.array([by[p][value] for by in results.values() if p in by])
        out[p] = (float(np.median(v)), float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)), len(v))
    return out
