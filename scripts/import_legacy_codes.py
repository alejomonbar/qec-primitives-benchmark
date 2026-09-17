"""Bring code-structure results of the MCM repository into this package's convention.

Device runs, ``<stamp>_<backend>_<sc|cc|qldpc>_..._nq<n>_depth<p>.json`` one per structure and depth,
are converted by ``qecbench.analysis.convert_legacy_code`` (the kind renamed, every derived number
recomputed from the samples, a logical ``CodePatch`` of the generated structure) and packed into one
run file per backend, family, kind and day:

    <dest>/<backend>/<family>/<kind>/<stamp>_<backend>_<family>_<kind>.json

where ``<stamp>`` is the day's first ``YYYYMMDD_HHMM``.

``--references`` reads noiseless *simulator* files of that repository instead. Above
``lrqaoa.MAX_STATEVECTOR_QUBITS`` data qubits there is no exact reference, so their sampled
``<H>`` and the histogram of their energies are kept in the reference store (``qecbench.references``)
as estimates, with the file and the number of shots they came from. Smaller problems are skipped: their reference is exact.

    python scripts/import_legacy_codes.py FILE_OR_GLOB [...] [-d DEST] [--anchor R0,C0 [--grid 12x10]]
                                          [--references [--method TEXT]] [--apply]

IBM runs store logical labels; ``--anchor`` rebuilds the physical surface-code patch they ran on (as the
position scan placed it) on a square-lattice chip of ``--grid`` rows x cols.

It only reports unless ``--apply`` is given. The original files are never modified.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from qecbench import codes, references  # noqa: E402
from qecbench.layout import surface_code_placements  # noqa: E402
from qecbench.analysis import (LEGACY_CODE_RE, convert_legacy_code, legacy_code_structure,  # noqa: E402
                               run_filename, save_run)
from qecbench.lrqaoa import MAX_STATEVECTOR_QUBITS, bitstring_energies  # noqa: E402


def import_runs(paths, dest, apply, anchor=None, grid=None):
    groups, skipped = defaultdict(list), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # "no noiseless reference" above 25 data qubits
        for path in paths:
            record = json.loads(path.read_text()) if path.exists() else None
            got = convert_legacy_code(record, path.name, placement=_placement(record, anchor, grid)) if record else None
            if got is None:
                skipped.append(path.name)
                continue
            stamp, record = got
            family = record["benchmark"]["instance"]["code"]["family"]
            key = (record["metadata"]["backend"], family, record["parameters"]["kind"], stamp[:8])
            groups[key].append((stamp, path, record))

    print(f"{len(paths)} file(s): {sum(map(len, groups.values()))} converted, {len(skipped)} skipped")
    for name in skipped:
        print("  skipped", name)
    for (backend, family, kind, day), items in sorted(groups.items()):
        stamp = min(s for s, _, _ in items)[:13]
        print(f"\n  {backend}/{family}/{kind}/{run_filename(backend, family, kind, stamp)}")
        for _, path, r in sorted(items, key=lambda t: (t[2]["parameters"]["num_data_qubits"], t[2]["parameters"]["depth"])):
            b = r["benchmark"]
            print(f"    {b['instance']['code']['name']:>11} p={r['parameters']['depth']:<3d} shots={r['parameters']['shots']:<4d} "
                  f"r={b['r']:.4f} r_ideal={b['r_ideal']:.4f}  <- {path.name}")
    if not apply:
        print("\nreport only - re-run with --apply to write")
        return
    for (backend, family, kind, day), items in sorted(groups.items()):
        stamp = min(s for s, _, _ in items)[:13]
        records = [r for _, _, r in items]
        run = {"backend": backend, "structure": family, "kind": kind, "stamp": stamp,
               "simulated": records[0]["metadata"]["simulated"],
               "noise_model": records[0]["metadata"]["noise_model"],
               "shots": sorted({r["parameters"]["shots"] for r in records}),
               "depths": sorted({r["parameters"]["depth"] for r in records}),
               "codes": sorted({r["benchmark"]["instance"]["code"]["name"] for r in records}),
               "source": "Benchmarking-Mid-circuit-measurement/Data",
               "source_files": sorted(p.name for _, p, _ in items)}
        out = save_run(Path(dest) / backend / family / kind / run_filename(backend, family, kind, stamp), records, run=run)
        print("wrote", out)


def _placement(record, anchor, grid):
    """The physical surface-code patch at ``anchor`` on a ``rows x cols`` square lattice, for IBM runs."""
    if anchor is None:
        return None
    rows, cols = grid
    G = nx.grid_2d_graph(rows, cols)
    G = nx.relabel_nodes(G, {(r, c): r * cols + c for r, c in G.nodes})
    n = int(record["parameters"]["num_data_qubits"])
    code = codes.surface_code(int(round(n ** 0.5)))
    found = surface_code_placements(G, code, anchors=[anchor])
    return found[0] if found else None


def import_references(paths, apply, method=None):
    for path in paths:
        m = LEGACY_CODE_RE.match(path.name.replace("_chi256", ""))
        record = json.loads(path.read_text())
        params, ham = record["parameters"], record["hamiltonian"]
        n = int(params["num_data_qubits"])
        hamiltonian = dict(zip((tuple(c) for c in ham["hamiltonian_couplings"]),
                               ham.get("hamiltonian_weights") or [1] * len(ham["hamiltonian_couplings"])))
        structure = legacy_code_structure(m["family"], n, hamiltonian, m["code"]) if m else None
        if structure is None or n <= MAX_STATEVECTOR_QUBITS:
            print(f"  skipped {path.name}: " + ("exact reference exists" if structure else "not a known structure"))
            continue
        bitstrings = list(record["samples"])
        counts = np.array([record["samples"][b] for b in bitstrings], dtype=float)
        shots = int(counts.sum())
        levels, inverse = np.unique(np.round(bitstring_energies(bitstrings, structure.hamiltonian), 9), return_inverse=True)
        frequencies = np.bincount(inverse, weights=counts, minlength=len(levels)) / shots
        energy = float(frequencies @ levels)
        method_note = method or "sampled simulation"
        method_note += ", MPS bond dimension 256" if "chi256" in path.name else ""
        depth, delta = int(params["depth"]), float(params.get("delta") or 0.5)
        span = structure.optimal_energy() - structure.max_energy()
        print(f"  {structure.name:>11} p={depth:<3d} <H>={energy:.4f} (r={(energy - structure.max_energy()) / span:.4f}), "
              f"{shots} shots, {method_note}  <- {path.name}")
        if apply:
            references.store(structure.hamiltonian, n, depth, delta, energy, method_note, code=structure.name,
                             shots=shots, backend=record["metadata"].get("backend"), source_file=path.name,
                             source="Benchmarking-Mid-circuit-measurement/Data",
                             distribution=[[float(e), float(q)] for e, q in zip(levels, frequencies)])
    if not apply:
        print("\nreport only - re-run with --apply to write")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="files or glob patterns")
    parser.add_argument("-d", "--dest", default="data/results", help="where the tree is written")
    parser.add_argument("--references", action="store_true", help="store simulator files as noiseless references")
    parser.add_argument("--anchor", help="r0,c0 of the surface-code patch an IBM run used (its files store logical labels)")
    parser.add_argument("--grid", default="12x10", help="rows x cols of the square-lattice chip, for --anchor")
    parser.add_argument("--method", help="how the simulator files were produced, e.g. 'MPS simulation on an HPC system'")
    parser.add_argument("--apply", action="store_true", help="actually write (default: report only)")
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.sources for p in (glob.glob(pattern) or [pattern])})
    if args.references:
        import_references(paths, args.apply, args.method)
    else:
        anchor = tuple(int(v) for v in args.anchor.split(",")) if args.anchor else None
        import_runs(paths, args.dest, args.apply, anchor, tuple(int(v) for v in args.grid.split("x")))


if __name__ == "__main__":
    main()
