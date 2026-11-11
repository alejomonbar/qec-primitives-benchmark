"""Build ``site/data/results.json``, everything the results website draws, from ``data/results/``.

Three sections, each read with the package's own loaders so the numbers are the ones the notebooks show:

* **chains** - MCM (and direct) LR-QAOA on 1D chains of ``N`` spins, per device and run file. A run file can
  hold many placements of one length (every triplet of a chip): the site gets the median over them and the best.
* **qec** - the LR-QAOA benchmark on QEC check Hamiltonians (surface, colour, qLDPC codes), per device, code and
  frame (``Z``: kind ``mcm``; ``X``: ``mcm_x``; ``XZ``: ``mcm_xz_z`` + ``mcm_xz_x``), median and best over patches.
* **memory** - surface-code memory experiments, per device, sitting and code: ``P_L`` after ``R`` rounds for
  ``|0>_L`` (basis Z) and ``|+>_L`` (basis X), median over patches, and the logical error per round.

Every point also gets ``r_eff = (r - t) / (1 - t)`` with ``t`` = random + 3 sigma, so ``r_eff > 0`` means clearly
above random guessing even where the noiseless reference is unknown (large codes).

    python scripts/build_site_data.py            # writes site/data/results.json
    python -m http.server -d site 8000           # preview at http://localhost:8000
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qecbench import memory as mem  # noqa: E402
from qecbench.analysis import load_results  # noqa: E402

RESULTS = ROOT / "data" / "results"
REGISTRY = ROOT / "scripts" / "site_devices.json"
OUT = ROOT / "site" / "data" / "results.json"
FRAME_KINDS = {"mcm": "Z", "mcm_x": "X", "mcm_xz_z": "XZ"}      # folder -> frame (mcm_xz_x is read with mcm_xz_z)
SKIP_KINDS = {"mcm_xz_x", "xframe", "memory"}


def num(x, digits=5):
    """A JSON-safe float (NaN and inf become null)."""
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, digits)


def stamp_of(name):
    m = re.match(r"(\d{8})_(\d{4})", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name[:13]


def date_of(name):
    s = stamp_of(name)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def summary_point(p, s):
    thr = s["r_rand"] + 3 * s["r_rand_std"]
    return {"p": int(p), "r": num(s["r"]), "r_err": num(s.get("r_err")), "r_ideal": num(s.get("r_ideal")),
            "r_ovl": num(s.get("r_ovl")), "threshold": num(thr), "r_eff": num((s["r"] - thr) / (1 - thr))}


def aggregate(instances, kind_label, keep_all=False):
    """Many placements of one problem in one run -> per depth the median and the best placement.

    The best placement is the one with the highest mean ``r_eff`` over the depths they all share.
    """
    depths = sorted(set.intersection(*(set(by) for by in instances.values())))
    if not depths:
        depths = sorted({p for by in instances.values() for p in by})
    pts = {inst: [summary_point(p, by[p]) for p in depths if p in by] for inst, by in instances.items()}
    best = max(pts, key=lambda i: np.mean([q["r_eff"] for q in pts[i]]))
    median = []
    for p in depths:
        here = [summary_point(p, by[p]) for by in instances.values() if p in by]
        median.append({k: num(np.median([q[k] for q in here if q[k] is not None])) if any(q[k] is not None for q in here)
                       else None for k in ("r", "r_ideal", "r_ovl", "threshold", "r_eff")} | {"p": p})
    first = next(iter(instances.values()))
    shots = int(first[min(first)]["shots"])
    above = [q["p"] for q in pts[best] if q["r_eff"] is not None and q["r_eff"] > 0]
    every = {}
    if keep_all and len(instances) > 1:                 # every placement's r, for a distribution (triplets)
        every = {"every": {str(p): [num(by[p]["r"], 4) for by in instances.values() if p in by] for p in depths}}
    return {"kind": kind_label, "placements": len(instances), "shots": shots, "median": median, "best": pts[best], **every,
            "best_qubits": sorted(int(q) for q in getattr(best, "qubits", ()))[:60],
            "deepest_p_above_random": max(above) if above else None,
            "best_r_eff": num(max(q["r_eff"] for q in pts[best]))}


def files_in(folder):
    return sorted(p.name for p in folder.glob("*.json")) if folder.is_dir() else []


def runs_of(backend, kind, folder, select):
    """``{file: {instance: {depth: summary}}}`` for the instances ``select`` keeps. Files that each hold a single
    depth of an instance (campaigns run depth by depth) are merged into one, dated by the newest."""
    per_file = {}
    for name in files_in(folder):
        names = {name, name.replace("_mcm_xz_z.json", "_mcm_xz_x.json")}     # the XZ frame's two halves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = load_results(RESULTS, backend, kind=kind, files=names)
        got = {i: by for i, by in got.items() if select(i)}
        if got:
            per_file[name] = got
    single = per_file and all(len(by) == 1 for got in per_file.values() for by in got.values())
    if single and len(per_file) > 1:
        merged = {}
        for name in sorted(per_file):                      # newest file last: it wins a repeated depth
            for inst, by in per_file[name].items():
                merged.setdefault(inst, {}).update(by)
        return {max(per_file): merged}
    return per_file


# ----------------------------------------------------------------------------------------------------------
def build_chains(backends):
    out = []
    for b in backends:
        for kind in ("mcm", "direct"):
            folder = RESULTS / b / "chain" / kind
            for name, got in runs_of(b, kind, folder, lambda i: getattr(i, "family", None) in ("chain", "direct")).items():
                by_n = {}
                for inst, by in got.items():
                    by_n.setdefault(inst.n_data, {})[inst] = by
                for n, insts in by_n.items():
                    run = aggregate(insts, kind, keep_all=(n == 2))
                    out.append({"id": f"{b}/{stamp_of(name)}/{kind}/{n}", "device": b, "nq": n, "date": date_of(name),
                                "file": name, **run})
    return out


def build_qec(backends):
    out = []
    for b in backends:
        for structure in ("surface_code", "color_code", "qldpc"):
            base = RESULTS / b / structure
            if not base.is_dir():
                continue
            for folder in sorted(p for p in base.iterdir() if p.is_dir() and p.name not in SKIP_KINDS):
                kind = folder.name
                frame = FRAME_KINDS.get(kind)
                if kind == "direct":
                    frame, load_kind = "Z", "direct"
                elif frame is None:
                    continue
                else:
                    load_kind = "mcm_xz" if kind == "mcm_xz_z" else kind
                for name, got in runs_of(b, load_kind, folder, lambda i: getattr(i, "family", None) == "code").items():
                    by_code = {}
                    for inst, by in got.items():
                        by_code.setdefault(inst.code.name, {})[inst] = by
                    for code, insts in by_code.items():
                        inst0 = next(iter(insts))
                        run = aggregate(insts, "direct" if kind == "direct" else "mcm")
                        out.append({"id": f"{b}/{stamp_of(name)}/{code}/{frame}/{run['kind']}", "device": b,
                                    "structure": structure, "code": code, "n_data": inst0.n_data,
                                    "checks": inst0.code.n_checks, "frame": frame, "date": date_of(name),
                                    "sitting": stamp_of(name), "file": name, "logical": bool(inst0.logical), **run})
    return out


def build_memory(backends):
    out = []
    for b in backends:
        folder = RESULTS / b / "surface_code" / "memory"
        for name in files_in(folder):
            got = mem.load_results(RESULTS, b, files={name})
            if not got:
                continue
            record = json.loads((folder / name).read_text())
            history = (record.get("run") or {}).get("history", "full")
            by_code = {}
            for patch, by_basis in got.items():
                by_code.setdefault(patch.code.name, {})[patch] = by_basis
            for code, patches in by_code.items():
                entry = {"id": f"{b}/{stamp_of(name)}/{code}", "device": b, "code": code, "date": date_of(name),
                         "sitting": stamp_of(name), "file": name, "patches": len(patches), "history": history,
                         "bases": {}}
                for basis in ("Z", "X"):
                    rs = sorted(set.intersection(*(set(v.get(basis, {})) for v in patches.values())))
                    if not rs:
                        continue
                    rates = np.array([[patches[q][basis][R]["rate"] for R in rs] for q in patches])
                    med = np.median(rates, axis=0)
                    fit = mem.error_per_round(rs, med) if len(rs) >= 2 else (None, None, None)
                    best = int(np.argmin(rates.mean(axis=1)))
                    entry["bases"][basis] = {
                        "R": rs, "median": [num(v) for v in med], "q25": [num(v) for v in np.percentile(rates, 25, axis=0)],
                        "q75": [num(v) for v in np.percentile(rates, 75, axis=0)], "best": [num(v) for v in rates[best]],
                        "eps_round": num(fit[0]), "shots": int(next(iter(patches.values()))[basis][rs[0]]["shots"])}
                if entry["bases"]:
                    out.append(entry)
    return out


def main():
    registry = json.loads(REGISTRY.read_text())
    devices = {k: v for k, v in registry["devices"].items()}
    backends = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and p.name not in registry.get("exclude", []))
    for b in backends:
        if b not in devices:
            print(f"warning: {b} is not in {REGISTRY.name}; listed as a QPU of unknown vendor")
            devices[b] = {"name": b, "vendor": "unknown", "kind": "qpu", "modality": "unknown"}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    data = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit,
            "devices": {b: devices[b] for b in backends},
            "chains": build_chains(backends), "qec": build_qec(backends), "memory": build_memory(backends)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    # the same data as a script, so the page also works opened straight from disk (file://), where a
    # browser refuses to fetch results.json
    OUT.with_suffix(".js").write_text("window.QEC_RESULTS = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"{len(data['chains'])} chain runs, {len(data['qec'])} QEC runs, {len(data['memory'])} memory runs "
          f"on {len(backends)} devices -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
