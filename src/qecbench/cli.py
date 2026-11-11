"""Command line:  python -m qecbench <command> ...

  run CONFIG.toml [overrides]   build, check and submit one experiment (or rehearse it locally)
  run CONFIG.toml --dry-run     build and check only: instances, circuits, cost; nothing is sent
  status [MANIFEST ...]         job status of submitted runs   (no argument: every run not yet collected)
  collect [MANIFEST ...]        fetch finished jobs into data/results/   (no argument: every run not collected)
  discard MANIFEST [...]        remove a run's results from data/results/ and stop tracking it

A run with a cancelled job is dropped by ``collect``: its result files are removed (the jobs that did finish
were part of an experiment that was stopped) and it is marked ``cancelled`` in the run list. A run whose jobs
failed with an error keeps what finished, is marked ``incomplete``, and is not waited on any longer.

A config is a short TOML file (examples in ``configs/``). ``experiment`` picks one of the three experiments:

* ``chain``  - LR-QAOA through the MCM gadget on 1D chains of ``nq`` spins (a list is a sweep), with the
  ancilla-free direct reference on the same qubits (``direct = true``);
* ``qec``    - LR-QAOA on the check Hamiltonian of a code (``code``, ``d``), in the ``frames`` Z, X and XZ;
* ``memory`` - the surface-code memory experiment in the ``bases`` Z (``|0>_L``) and X (``|+>_L``) over
  ``rounds``.

The device decides the vendor: ``ibm_*`` (Qiskit Runtime, a saved account), ``iqm_*`` (Amazon Braket),
``Helios-1``/``Helios-1E``/``H2-*`` (Quantinuum Nexus), ``noisy_simulator`` (no account at all). ``local =
true`` (or ``--local``) runs the same plan on Aer instead and files it as ``<device>_sim``; a local run is
collected straight away. Every submitted run is listed in ``data/manifests/runs.json`` so ``status`` and
``collect`` find it without arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = ("chain", "qec", "memory")
LEDGER = "runs.json"
# the fixed patch set the ibm_phoenix LR-QAOA and memory runs share (the 2026-09-14 scan)
SCAN_ANCHORS = {"ibm_phoenix": {3: [(7, 2), (7, 3), (6, 3), (0, 3), (0, 4), (7, 5), (6, 4), (2, 5), (3, 7), (7, 7),
                                    (3, 2)],
                                5: [(0, 5), (0, 4), (3, 5), (1, 5), (1, 4), (2, 4), (3, 4), (2, 5)]}}
FAMILIES = {"surface": "surface_code", "color": "color_code", "colour": "color_code", "qldpc": "qldpc"}
DEFAULTS = {
    # one chain per length, four lengths, four depths: 16 circuits catch the peak (p = 3), the decay and where
    # it reaches random (p = 20-40) on every IBM chip measured so far. chains_per_length = 0 covers every
    # ancilla-capable qubit instead (a chip survey, many more circuits).
    "chain": dict(nq=[10, 20, 40, 60], ps=[3, 10, 20, 40], shots=1000, delta=0.5, direct=False, buffer=False,
                  chains_per_length=1),
    "qec": dict(code="surface", d=[3], ps=[1, 2, 3, 5, 7, 10], shots=1000, delta=0.5, frames=["Z"], direct=False,
                positions=11, max_coupler_error=0.30, max_readout_error=0.30),
    "memory": dict(code="surface", d=[3], rounds=[1, 2, 3, 5, 7, 10], bases=["Z", "X"], shots=4000,
                   p_2q_model=3e-3, positions=11, max_coupler_error=0.30, max_readout_error=0.30),
}


# ======================================================================================
# Config
# ======================================================================================
def load_config(path=None, **overrides):
    """The TOML file, then any non-None override, then the experiment's defaults."""
    cfg = {}
    if path:
        try:
            import tomllib
        except ModuleNotFoundError:                 # Python 3.10
            import tomli as tomllib
        cfg = tomllib.loads(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    exp = cfg.get("experiment")
    if exp not in EXPERIMENTS:
        sys.exit(f"experiment must be one of {EXPERIMENTS}, not {exp!r}")
    if not cfg.get("backend"):
        sys.exit("the config needs a backend (e.g. ibm_phoenix, iqm_emerald, Helios-1, noisy_simulator)")
    out = {**DEFAULTS[exp], **cfg}
    for key in ("nq", "d", "ps", "rounds", "frames", "bases"):
        if key in out and not isinstance(out[key], list):
            out[key] = [out[key]]
    out.setdefault("provider", {})
    out.setdefault("local", False)
    out.setdefault("data_dir", "data")
    return out


def vendor_of(name):
    if name == "noisy_simulator":
        return "sim"
    if name.startswith("ibm_"):
        return "ibm"
    if name.startswith("iqm_"):
        return "iqm"
    if name.startswith(("Helios", "H2")):
        return "quantinuum"
    sys.exit(f"unknown device {name!r}: expected ibm_*, iqm_*, Helios-1(E), H2-* or noisy_simulator")


def make_backend(cfg, local=None, calibrate=True):
    """The backend of a config, and its calibration (None when there is none, or ``calibrate=False``)."""
    name, prov = cfg["backend"], cfg["provider"]
    local = cfg["local"] if local is None else local
    vendor = vendor_of(name)
    data = Path(cfg["data_dir"])
    if vendor == "sim":
        from .backends import SimBackend

        grid = cfg["experiment"] != "chain"            # codes need a square lattice
        return SimBackend(topology=prov.get("topology", "grid" if grid else "heavy_hex"),
                          qubits=prov.get("qubits", 121 if grid else 54), seed=prov.get("seed", 1)), None
    if vendor == "ibm":
        from .backends import IBMBackend

        backend = IBMBackend(name, account=prov.get("account"), local=local, seed=prov.get("seed", 1),
                             optimization_level=prov.get("optimization_level", 1),
                             dynamical_decoupling=prov.get("dynamical_decoupling", False),
                             max_circuits_per_job=prov.get("max_circuits_per_job", 8))
        return backend, (backend.calibration(save_dir=data / "calibration") if calibrate else None)
    if vendor == "iqm":
        from .backends import IQMBackend

        backend = IQMBackend(name, local=local, seed=prov.get("seed", 1))
        if not calibrate:
            return backend, None
        try:
            cal = backend.calibration(save_dir=data / "calibration")
        except Exception as exc:                        # no AWS access: a stored snapshot, if any
            stored = sorted((data / "calibration").glob(f"{name}*/*_calibration.json"))
            cal = json.loads(stored[-1].read_text()) if stored else None
            print(f"no live calibration ({type(exc).__name__}); "
                  + (f"using {stored[-1].name}" if stored else "none stored"))
        return backend, cal
    from .backends import QuantinuumBackend

    return QuantinuumBackend(name, local=local, project=prov.get("project", "Helios-Samples"),
                             cost_margin=prov.get("cost_margin", 3.0), seed=prov.get("seed", 7)), None


# ======================================================================================
# Instances
# ======================================================================================
def chain_instances_for(cfg, backend, cal):
    from . import Direct
    from .layout import restrict_to_groups, select_chains

    kinds = ("mcm", "direct") if cfg["direct"] else ("mcm",)
    if vendor_of(cfg["backend"]) == "quantinuum":
        from .backends import chain_instances

        return chain_instances(cfg["nq"], kinds)
    G = backend.coupling_graph()
    if vendor_of(cfg["backend"]) == "iqm":
        G = restrict_to_groups(G, backend.feedforward_groups())
    cost = (lambda c: backend.error_budget(c, cal)) if cal else None
    allowed = cfg.get("allowed_qubits")
    if allowed is None and getattr(backend, "simulated", False) and vendor_of(cfg["backend"]) != "iqm":
        allowed = list(range(20))                  # Aer runs whole-chip circuits: keep a rehearsal small
        print("local simulation: chains on qubits 0-19 only (set allowed_qubits to change it)")
    import time

    out = []
    for n in cfg["nq"]:
        print(f"N = {n:>3}: searching chains of {2 * n - 1} qubits ...", end=" ", flush=True)
        t0 = time.monotonic()
        chains = select_chains(G, n, allowed_qubits=allowed, cost=cost, buffer=cfg["buffer"],
                               time_limit=cfg.get("search_seconds", 120))
        if not chains:
            print(f"none found in {time.monotonic() - t0:.0f} s: no {2 * n - 1}-qubit path for a chain of {n} spins "
                  f"on {backend.name} ({G.number_of_nodes()} qubits); this length is left out")
            continue
        print(f"done in {time.monotonic() - t0:.0f} s")
        keep = cfg.get("chains_per_length", 1)
        if keep and len(chains) > keep:                 # the best-calibrated ones, or the first found
            chains = sorted(chains, key=cost)[:keep] if cost else chains[:keep]
        direct = sorted({Direct.from_chain(c) for c in chains}) if cfg["direct"] else []
        print(f"N = {n:>3}: {len(chains)} chains" + (f" + {len(direct)} direct references" if direct else ""))
        out += chains + direct
    return out


def code_structures(cfg):
    from . import codes

    family = FAMILIES.get(cfg["code"], cfg["code"])
    return [codes.build(family, d) for d in cfg["d"]]


def placements_for(cfg, backend, cal, code):
    """Surface-code patches on a square-lattice chip: the fixed patch set, given anchors, or ``positions``
    spread over the calibrated error budget. Patches on hardware over the thresholds are left out, with why."""
    from .layout import square_lattice_coordinates, surface_code_placements, spread_selection

    G = backend.coupling_graph()
    if square_lattice_coordinates(G) is None:
        sys.exit(f"{backend.name}: couplers do not form a square lattice, so surface-code patches cannot be placed")
    d = int(code.info.get("distance") or round(code.n_data ** 0.5))
    anchors = cfg.get("anchors")
    if anchors == "scan" or (anchors is None and cfg["backend"] in SCAN_ANCHORS):
        anchors = SCAN_ANCHORS.get(cfg["backend"], {}).get(d)
    elif isinstance(anchors, dict):
        anchors = anchors.get(str(d)) or anchors.get(d)
    anchors = [tuple(a) for a in anchors] if anchors else None
    found = surface_code_placements(G, code, anchors=anchors)
    if cal is not None and hasattr(backend, "flag_instances") and vendor_of(cfg["backend"]) == "ibm":
        flagged = dict(backend.flag_instances(found, cal, max_2q_error=cfg["max_coupler_error"],
                                              max_readout_error=cfg["max_readout_error"]))
        for patch, why in flagged.items():
            print(f"   left out {patch}: {why}")
        found = [p for p in found if p not in flagged]
    if anchors is None and found:
        scores = [backend.error_budget(p, cal) for p in found] if cal else list(range(len(found)))
        found = [p for p, _ in spread_selection(found, scores, cfg["positions"])]
    print(f"d = {d}: {len(found)} patch(es)" + (" from the fixed patch set" if anchors else ""))
    if not found:
        print(f"   no usable placement of d = {d} on {backend.name}; d = {d} is left out of the run")
    return found


# ======================================================================================
# run / status / collect
# ======================================================================================
def run(cfg, dry_run=False):
    exp, data = cfg["experiment"], Path(cfg["data_dir"])
    vendor = vendor_of(cfg["backend"])
    if exp == "qec" and vendor == "iqm":
        sys.exit("QEC structures cannot run through the MCM gadget on IQM: a data qubit would be "
                 "corrected by several ancillas, and IQM allows one controller per qubit")
    if exp == "memory" and vendor == "quantinuum":
        sys.exit("the memory experiment is built for fixed-coupling chips (IBM, IQM) and the simulator")
    backend, cal = make_backend(cfg)
    print(f"{exp} on {backend.name}" + (" (local rehearsal on Aer)" if getattr(backend, "simulated", False)
                                         or getattr(backend, "local", False) else ""))

    if exp == "memory":
        from . import memory as mem

        patches = [p for code in code_structures(cfg) for p in placements_for(cfg, backend, cal, code)]
        if not patches:
            print("nothing to run: no usable patch")
            return None
        plan = mem.plan(backend, patches, rounds=cfg["rounds"], bases=cfg["bases"], shots=cfg["shots"],
                        p_2q_model=cfg["p_2q_model"])
        mem.print_plan(plan)
        if dry_run:
            return None
        if vendor == "sim":
            noise = mem.uniform_noise_model(**backend.noise_params)
        elif cal is not None:
            noise = lambda patch: mem.calibrated_noise_model(patch, cal)       # noqa: E731
        else:
            noise = mem.uniform_noise_model(p_1q=1e-3, p_2q=1e-2, readout=2e-2)
        manifest = mem.submit(plan, backend, manifest_dir=data / "manifests", noise_model=noise, seed=7)
    else:
        from .experiment import build_plan, print_plan, submit

        if exp == "chain":
            instances = chain_instances_for(cfg, backend, cal)
            frames = ("Z",)
        elif vendor == "quantinuum":
            from .backends import code_instances

            kinds = ("mcm", "direct") if cfg["direct"] else ("mcm",)
            instances = [i for s in code_structures(cfg) for i in code_instances(s, kinds)]
            frames = tuple(cfg["frames"])
        else:
            instances = [p for code in code_structures(cfg) for p in placements_for(cfg, backend, cal, code)]
            frames = tuple(cfg["frames"])
        if not instances:
            print("nothing to run: no usable instance")
            return None
        extra = {"max_per_batch": 1} if exp == "qec" and vendor != "quantinuum" else {}
        if frames != ("Z",):
            extra["frames"] = frames
        plan = build_plan(backend, instances, depths=cfg["ps"], shots=cfg["shots"], delta=cfg["delta"], **extra)
        if vendor == "quantinuum" and not backend.simulated and not dry_run:
            if backend.quote(plan, wait=True) is None:
                print_plan(plan, backend)
                print("the Nexus cost estimate is not back yet; nothing was sent - run the same command again")
                return None
        print_plan(plan, backend)
        if dry_run:
            return None
        manifest = submit(plan, backend, manifest_dir=data / "manifests")

    _ledger_add(data, manifest, cfg, backend.name)
    if getattr(backend, "simulated", False) or getattr(backend, "local", False) or vendor == "sim":
        collect([Path(manifest)], data)                 # nothing to wait for
    else:
        print(f"\nsubmitted. Check with   python -m qecbench status\n"
              f"collect with             python -m qecbench collect")
    return manifest


def _ledger(data):
    path = Path(data) / "manifests" / LEDGER
    return path, (json.loads(path.read_text()) if path.exists() else [])


def _ledger_add(data, manifest, cfg, name):
    path, rows = _ledger(data)
    rows.append({"manifest": str(manifest), "experiment": cfg["experiment"], "backend": name,
                 "config": {k: v for k, v in cfg.items() if k != "data_dir"},
                 "submitted": datetime.now().isoformat(timespec="seconds"), "status": "submitted"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, default=str))


def _entry(data, manifest):
    _, rows = _ledger(data)
    return next((r for r in rows if Path(r["manifest"]).name == Path(manifest).name), None)


def _manifests(files, data):
    if files:
        return [Path(f) for f in files]
    _, rows = _ledger(data)
    return [Path(r["manifest"]) for r in rows if r["status"] == "submitted"]


def _mark(data, manifest, state):
    path, rows = _ledger(data)
    for r in rows:
        if Path(r["manifest"]).name == Path(manifest).name:
            r["status"] = state
    if rows:
        path.write_text(json.dumps(rows, indent=1, default=str))


def run_files(data, manifest):
    """The result files a manifest's harvest wrote: named by its creation time, and naming it inside."""
    record = json.loads(Path(manifest).read_text())
    stamp = datetime.fromisoformat(record["created"]).strftime("%Y%m%d_%H%M%S")
    out = []
    for path in sorted((Path(data) / "results" / record["backend"]).rglob(f"{stamp}_*.json")):
        try:
            if json.loads(path.read_text()).get("run", {}).get("manifest") == Path(manifest).name:
                out.append(path)
        except (ValueError, OSError):
            continue
    return out


def discard(files, data, reason="discarded"):
    """Remove the result files of these runs and mark them in the run list; the manifests are kept as a record."""
    for m in [Path(f) for f in files]:
        removed = run_files(data, m)
        for path in removed:
            path.unlink()
        _mark(data, m, reason)
        print(f"{m.name}: {reason}; " + (f"removed {len(removed)} result file(s): "
                                         + ", ".join(p.name for p in removed) if removed else "no result files to remove"))


def _backend_for(data, manifest):
    """The backend that can fetch a manifest's jobs: rebuilt from the config it was run with."""
    entry = _entry(data, manifest)
    record = json.loads(Path(manifest).read_text())
    if entry is None:
        name = record["backend"].removesuffix("_sim")
        exp = "memory" if record.get("experiment") == "memory" else "chain"
        cfg = load_config(None, experiment=exp, backend=name, data_dir=str(data))
    else:
        cfg = {**entry["config"], "data_dir": str(data)}
    local = record.get("simulated", False) or record["backend"].endswith("_sim") or all(
        "counts" in j for j in record.get("jobs", []))
    if local and vendor_of(cfg["backend"]) == "ibm":
        # a finished local run holds its counts; no account is needed to read them back
        from .backends import AerBackend

        backend = AerBackend(name=record["backend"])
        backend.vendor = "ibm"
        return backend, record
    return make_backend(cfg, local=local, calibrate=False)[0], record


def job_states(backend, record):
    """``[(job id, state)]`` of a manifest's jobs; a local job is always ``DONE``."""
    out = []
    for job in record.get("jobs", []):
        jid = job.get("job_id")
        if "counts" in job:
            out.append((jid, "DONE"))
            continue
        try:
            if getattr(backend, "vendor", None) == "iqm":
                from braket.aws import AwsQuantumTask

                state = AwsQuantumTask(arn=jid).state()
            elif getattr(backend, "vendor", None) == "quantinuum":
                qnx = backend._qnx()
                status = qnx.jobs.status(qnx.jobs.get(id=jid))
                state = str(getattr(status.status, "value", status.status))
            else:
                state = str(backend.service.job(jid).status())
        except Exception as exc:                    # report it and keep going
            state = f"unknown ({type(exc).__name__}: {exc})"
        out.append((jid, state.replace("JobStatus.", "")))
    return out


FINISHED = {"DONE", "COMPLETED"}
CANCELLED = {"CANCELLED", "CANCELED"}
FAILED = {"ERROR", "FAILED"}


def status(files, data):
    manifests = _manifests(files, data)
    if not manifests:
        print("no submitted run waiting to be collected")
    for m in manifests:
        backend, record = _backend_for(data, m)
        states = job_states(backend, record)
        done = sum(s in FINISHED for _, s in states)
        cancelled = sum(s in CANCELLED for _, s in states)
        failed = sum(s in FAILED for _, s in states)
        print(f"{m.name}: {done}/{len(states)} jobs finished"
              + (f", {cancelled} cancelled - collect will drop this run and its results" if cancelled else "")
              + (f", {failed} failed - collect will keep what finished" if failed and not cancelled else ""))
        for jid, s in states:
            if s not in FINISHED | CANCELLED:
                print(f"   {jid}: {s}")


def collect(files, data):
    manifests = _manifests(files, data)
    if not manifests:
        print("no submitted run waiting to be collected")
    for m in manifests:
        backend, record = _backend_for(data, m)
        states = job_states(backend, record)
        if any(s in CANCELLED for _, s in states):
            n = sum(s in CANCELLED for _, s in states)
            discard([m], data, reason="cancelled")
            print(f"   ({n} of {len(states)} jobs were cancelled, so the run is not kept)")
            continue
        print(f"collecting {m.name}")
        if record.get("experiment") == "memory":
            from . import memory as mem

            mem.harvest(m, backend, data_dir=Path(data) / "results")
        else:
            from .experiment import harvest

            harvest(m, backend, data_dir=Path(data) / "results")
        if all(s in FINISHED for _, s in states):
            _mark(data, m, "collected")
        elif all(s in FINISHED | FAILED for _, s in states):
            _mark(data, m, "incomplete")
            print(f"   {sum(s in FAILED for _, s in states)} job(s) failed; the finished ones are kept, "
                  "and the run is no longer waited on")


# ======================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m qecbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build, check and submit one experiment")
    r.add_argument("config", nargs="?", help="TOML file (see configs/)")
    r.add_argument("--experiment", choices=EXPERIMENTS)
    r.add_argument("--backend")
    r.add_argument("--nq", type=int, nargs="+", help="chain: spins per chain (several = a sweep)")
    r.add_argument("--code", help="qec / memory: surface, color or qldpc")
    r.add_argument("--d", nargs="+", help="qec / memory: code distances (or BB18 ... for qldpc)")
    r.add_argument("--frames", nargs="+", choices=["Z", "X", "XZ"])
    r.add_argument("--ps", type=int, nargs="+", help="LR-QAOA layers")
    r.add_argument("--rounds", type=int, nargs="+", help="memory: rounds of syndrome extraction")
    r.add_argument("--bases", nargs="+", choices=["Z", "X"], help="memory: Z = |0>_L, X = |+>_L")
    r.add_argument("--shots", type=int)
    r.add_argument("--local", action="store_true", default=None, help="rehearse on Aer; nothing is sent")
    r.add_argument("--data-dir")
    r.add_argument("--dry-run", action="store_true", help="build and check only, print the plan and its cost")
    for name, helptext in (("status", "job status of submitted runs"), ("collect", "fetch finished jobs"),
                           ("discard", "remove a run's results and stop tracking it")):
        c = sub.add_parser(name, help=helptext)
        c.add_argument("manifests", nargs="+" if name == "discard" else "*",
                       help="manifest files" + ("" if name == "discard" else " (default: every run still waiting)"))
        c.add_argument("--data-dir", default="data")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        d = [int(x) if str(x).isdigit() else x for x in a.d] if a.d else None
        cfg = load_config(a.config, experiment=a.experiment, backend=a.backend, nq=a.nq, code=a.code, d=d,
                          frames=a.frames, ps=a.ps, rounds=a.rounds, bases=a.bases, shots=a.shots,
                          local=a.local, data_dir=a.data_dir)
        run(cfg, dry_run=a.dry_run)
    elif a.cmd == "status":
        status(a.manifests, a.data_dir)
    elif a.cmd == "discard":
        discard(a.manifests, a.data_dir)
    else:
        collect(a.manifests, a.data_dir)


if __name__ == "__main__":
    main()
