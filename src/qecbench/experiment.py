"""Plan -> submit (manifest) -> harvest, identical for every vendor.

* ``build_plan`` packs the instances into circuits, validates every batch against the
  vendor rules, builds (and for IBM transpiles) every circuit and prices the lot.
* ``submit`` sends jobs of at most ``backend.max_circuits_per_job`` circuits and writes the
  manifest **after every job**, so an interrupted submission still knows its job ids. Whether
  that means the device or a local simulation is the backend's choice (``local=True``,
  ``AerBackend``, ``SimBackend``), so there is exactly one switch. ``print_plan`` shows what
  would run and what it costs before anything is sent.
* ``harvest`` reads a manifest - possibly days later, in another session - and writes one
  analysed result file per instance and depth.  Safe to re-run.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .analysis import (instance_from_record, make_record, public_job_id, record_key, records_in, result_dir,
                       run_filename, save_run, short_path)
from .circuits import FRAMES, result_kind
from .layout import pack
from .primitives import from_dict


def build_plan(backend, instances, depths, shots, delta=0.5, buffer=False,
               max_per_batch=None, strategy=None, frames=("Z",)):
    """Pack, validate, build and price every circuit of an experiment.

    The circuit an instance gets follows from what it is: a ``Chain`` runs the
    mid-circuit-measurement gadget (kind ``mcm``), a ``Direct`` runs the same Ising chain
    straight on its couplers (kind ``direct``).  Pass both - e.g. ``chains +
    [Direct.from_chain(c) for c in chains]`` - and both go into the same jobs, so the
    reference sees the same calibration as the benchmark.  Each kind is packed separately,
    since their instances need different circuits.

    ``frames=("Z", "X")`` also builds every ``mcm`` batch in the X frame (``circuits.build_dynamic``), right after
    its Z-frame circuit so the two share a job; those results are filed as kind ``mcm_x``. ``direct`` batches are
    built in the Z frame only.
    """
    frames = tuple(dict.fromkeys(frames))
    if not frames or set(frames) - set(FRAMES):
        raise ValueError(f"frames must be taken from {FRAMES}, not {frames}")
    unsupported = set(frames) - set(getattr(backend, "frames", ("Z",)))
    if unsupported:
        raise ValueError(f"{backend.name} builds frames {getattr(backend, 'frames', ('Z',))}, not {sorted(unsupported)}")
    instances = sorted(set(instances))
    if not instances:
        raise ValueError("no instances to run")
    kinds = sorted({inst.kind for inst in instances})
    bad = [k for k in kinds if k not in backend.kinds]
    if bad:
        raise ValueError(f"{backend.name} does not run kinds {bad}; it runs {backend.kinds}")
    G = backend.coupling_graph()
    batches = {}
    for kind in kinds:
        batches[kind] = pack([i for i in instances if i.kind == kind], G, buffer=buffer,
                             max_per_batch=max_per_batch,
                             strategies=[strategy] if strategy else None)
        for i, batch in enumerate(batches[kind]):
            problems = backend.validate(batch, kind)
            if problems:
                raise ValueError(f"{backend.name} rejects batch {i} ({kind}):\n  " + "\n  ".join(problems[:10])
                                 + (f"\n  ... {len(problems) - 10} more" if len(problems) > 10 else ""))
    tasks, circuits = [], []
    for depth in depths:
        for kind in kinds:
            for i, batch in enumerate(batches[kind]):
                for frame in (frames if kind == "mcm" else ("Z",)):
                    qc = (backend.build(batch, depth, delta, kind) if frame == "Z"
                          else backend.build(batch, depth, delta, kind, frame=frame))
                    problems = backend.check_built(qc, batch)
                    if problems:
                        raise ValueError(f"batch {i}, depth {depth}, {kind} ({frame} frame): " + "; ".join(problems[:5]))
                    tasks.append({"kind": result_kind(kind, frame), "frame": frame, "depth": depth, "batch": i,
                                  "instances": batch})
                    circuits.append(qc)
    plan = {"backend": backend.name, "vendor": backend.vendor, "instances": instances,
            "batches": batches, "depths": list(depths), "shots": shots, "delta": delta,
            "kinds": kinds, "frames": list(frames), "buffer": buffer, "max_per_batch": max_per_batch,
            "tasks": tasks, "circuits": circuits}
    plan["estimate"] = backend.estimate(plan)
    return plan


def print_plan(plan, backend=None):
    """What will run, in how many circuits and jobs, and what it costs - before submitting."""
    inst = plan["instances"]
    n = len(plan["circuits"])
    per_job = min(backend.max_circuits_per_job if backend else 1, n) or 1
    print(f"Backend    : {plan['backend']} ({plan['vendor']})")
    print(f"Sweep      : depths {plan['depths']}, {plan['shots']} shots per circuit, delta {plan['delta']}")
    print(f"Instances  : {len(inst)} in total, "
          f"{len({a for i in inst for a in i.ancillas})} qubits benchmarked as ancilla")
    for kind in plan["kinds"]:
        of_kind = [i for i in inst if i.kind == kind]
        fam = Counter((i.code.name if i.family == "code" else i.family, i.n_data) for i in of_kind)
        batches = plan["batches"][kind]
        print(f"  {kind:<7}: " + ", ".join(f"{c} x {f} of {k} data qubits"
                                           for (f, k), c in sorted(fam.items())))
        print(f"  {'':<7}  {len(batches)} circuit(s) per depth, running "
              f"{' + '.join(str(len(b)) for b in batches)} instances in parallel")
    print(f"Circuits   : {n} = {len(plan['depths'])} depths x {n // len(plan['depths'])} circuits per depth"
          + (f" (frames {', '.join(plan['frames'])})" if plan.get("frames", ["Z"]) != ["Z"] else ""))
    print(f"Jobs       : {-(-n // per_job)}, up to {per_job} circuit(s) each")
    est = plan.get("estimate")
    if not est:
        return
    if est.get("total") is None:
        print(f"Cost       : pending - {est['unit']} estimate {est.get('job_id')} has not finished; "
              f"quote the plan again to read it")
        return
    header = f"\n{'depth':>6} {'kind':>8} {'circuits':>9} {'instances':>10} {est['unit']:>10}"
    print(header)
    print("-" * (len(header) - 1))
    for depth in plan["depths"]:
        for kind in sorted({t["kind"] for t in plan["tasks"]}):
            sel = [i for i, t in enumerate(plan["tasks"])
                   if t["depth"] == depth and t["kind"] == kind]
            if not sel:
                continue
            print(f"{depth:>6} {kind:>8} {len(sel):>9} "
                  f"{sum(len(plan['tasks'][i]['instances']) for i in sel):>10} "
                  f"{sum(est['per_circuit'][i] for i in sel):>10.2f}")
    print("-" * (len(header) - 1))
    print(f"{'TOTAL':>15} {n:>9} {'':>10} {est['total']:>10.2f}"
          + ("" if est.get("usd") is None else f"  = ${est['usd']:.2f}"))
    for note in est.get("notes", []):
        print(f"  note: {note}")


def submit(plan, backend, manifest_dir="data/manifests", label=None):
    n, size = len(plan["circuits"]), backend.max_circuits_per_job
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    families = sorted({i.code.name if i.family == "code" else f"{i.family}{i.n_data}" for i in plan["instances"]})
    name = f"{stamp}_{backend.name}_{'-'.join(families)}{'_' + label if label else ''}"
    path = Path(manifest_dir) / backend.name / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    est = plan.get("estimate") or {}
    if est and est.get("total") is None and hasattr(backend, "quote"):
        est = backend.quote(plan) or est          # look at the estimate job now rather than trust the plan
    if est and est.get("total") is None:
        raise RuntimeError(f"the {est['unit']} estimate {est.get('job_id')} is still running, and the per-program "
                           f"budgets come from it: submit again once it has finished. Nothing was sent.")
    manifest = {"backend": backend.name, "vendor": backend.vendor, "created": datetime.now().isoformat(),
                "shots": plan["shots"], "delta": plan["delta"], "depths": plan["depths"],
                "kinds": plan["kinds"], "frames": plan.get("frames", ["Z"]), "buffer": plan["buffer"],
                "max_per_batch": plan["max_per_batch"],
                "estimate": {k: est.get(k) for k in ("unit", "total", "usd")},
                "simulated": backend.simulated, "noise_model": backend.noise_description,
                "jobs": []}
    simulating = backend.simulated
    verb = "SIMULATING" if simulating else "SUBMITTING"
    # on a simulator the estimate is what the hardware would have cost, not what this run costs
    usd = "" if est.get("usd") is None else f" (${est['usd']:.2f})"
    cost = "" if not est else (f", which on the device would cost {est['total']:.2f} {est['unit']}{usd}"
                               if simulating else f", estimated {est['total']:.2f} {est['unit']}{usd}")
    print(f"{verb} {n} circuits to {backend.name} in {-(-n // size)} job(s)" + cost)
    for start in range(0, n, size):
        chunk = list(range(start, min(start + size, n)))
        record = {"job_id": None,
                  "tasks": [{"kind": plan["tasks"][i]["kind"], "frame": plan["tasks"][i].get("frame", "Z"),
                             "depth": plan["tasks"][i]["depth"],
                             "batch": plan["tasks"][i]["batch"], "index": k,
                             "instances": [inst.to_dict() for inst in plan["tasks"][i]["instances"]]}
                            for k, i in enumerate(chunk)]}
        record.update(backend.submit([plan["circuits"][i] for i in chunk], plan["shots"]))
        print(f"  job {len(manifest['jobs']) + 1}: {len(chunk)} circuit(s) -> {record['job_id']}")
        manifest["jobs"].append(record)
        path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {short_path(path)}")
    return str(path)


def harvest(manifest_path, backend, data_dir="data/results", overwrite=False):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("experiment") == "memory":
        raise ValueError(f"{Path(manifest_path).name} is a memory-experiment manifest: harvest it with "
                         f"qecbench.memory.harvest (benchmark_qec_memory.ipynb)")
    existing = {} if overwrite else _existing(data_dir, manifest["backend"])
    stamp = datetime.fromisoformat(manifest["created"]).strftime("%Y%m%d_%H%M%S")
    header = {k: manifest.get(k) for k in ("backend", "vendor", "created", "shots", "delta",
                                           "depths", "kinds", "frames", "buffer", "simulated",
                                           "noise_model", "estimate")}
    header["manifest"] = Path(manifest_path).name
    groups, skipped = {}, 0
    for record in manifest["jobs"]:
        if not record.get("job_id"):
            continue
        todo = [t for t in record["tasks"]
                if any((public_job_id(record["job_id"]), t["kind"], t["depth"], from_dict(s).identity) not in existing
                       for s in t["instances"])]
        if not todo:
            skipped += sum(len(t["instances"]) for t in record["tasks"])
            continue
        per_task = backend.fetch(record, todo)
        if per_task is None:
            continue
        for task, inst_counts in zip(todo, per_task):
            for spec, counts in zip(task["instances"], inst_counts):
                inst = from_dict(spec)
                key = (public_job_id(record["job_id"]), task["kind"], task["depth"], inst.identity)
                if key in existing:
                    skipped += 1
                    continue
                rec = make_record(counts, inst, depth=task["depth"], delta=manifest["delta"],
                                  backend_name=manifest["backend"], job_id=record["job_id"],
                                  kind=task["kind"], batch=task["batch"],
                                  batch_size=len(task["instances"]),
                                  simulated=manifest.get("simulated", False),
                                  noise=manifest.get("noise_model"),
                                  extra={"manifest": Path(manifest_path).name,
                                         **({"frame": task["frame"]} if task.get("frame", "Z") != "Z" else {})})
                groups.setdefault((inst.structure, task["kind"]), []).append(rec)
                existing[key] = True

    saved, written = [], 0
    for (structure, kind), records in sorted(groups.items()):
        folder = Path(data_dir) / manifest["backend"] / structure / kind
        saved.append(save_run(folder / run_filename(manifest["backend"], structure, kind, stamp),
                              records, run=header))
        written += len(records)
    print(f"{written} results written to {len(saved)} run file(s) under {short_path(data_dir)}, "
          f"{skipped} already there")
    return saved


def _existing(data_dir, backend_name):
    """``{(job, kind, depth, qubits): path}`` for results already on disk, any family.

    Keyed by the job as well as the instance, so a new campaign on the same qubits is
    harvested rather than mistaken for one already there.
    """
    out = {}
    for path in Path(data_dir).rglob("*.json"):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for record in records_in(document):
            if record.get("metadata", {}).get("backend") != backend_name:
                continue
            if instance_from_record(record) is not None:
                out[record_key(record)] = str(path)
    return out
