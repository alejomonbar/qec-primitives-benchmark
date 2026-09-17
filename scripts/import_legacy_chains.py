"""Bring 1D-chain results of the MCM repository into this package's convention.

The source files are ``<stamp>_<backend>_[1d*_]<normal|MCM>_nq<n>_depth<p>.json`` (chains) and
``<stamp>_<backend>_tri_<d1>_<a>_<d2>_mcm_nq2_depth<p>.json`` (triplets) and
``<stamp>_<backend>_nq2_depth<p>.json`` (early two-spin runs), one per instance and depth. Each is converted by ``qecbench.analysis.convert_legacy_chain`` (the kind
renamed, every derived number recomputed from the samples, logical ancilla labels where the device
reused qubits) and packed into one run file per submission:

    <dest>/<backend>/chain/<kind>/<stamp>_<backend>_chain_<kind>.json

    python scripts/import_legacy_chains.py FILE_OR_GLOB [...] [-d DEST] [--apply]

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

from qecbench.analysis import convert_legacy_chain, run_filename, save_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="files or glob patterns")
    parser.add_argument("-d", "--dest", default="data/results", help="where the tree is written")
    parser.add_argument("--apply", action="store_true", help="actually write (default: report only)")
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.sources for p in (glob.glob(pattern) or [pattern])})
    groups, skipped = defaultdict(list), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # MPS notice for chains past 25 spins
        for path in paths:
            got = convert_legacy_chain(json.loads(path.read_text()), path.name) if path.exists() else None
            if got is None:
                skipped.append(path.name)
                continue
            stamp, record = got
            key = (record["metadata"]["backend"], "chain", record["parameters"]["kind"], stamp)
            groups[key].append((path, record))

    print(f"{len(paths)} file(s): {sum(map(len, groups.values()))} converted, {len(skipped)} skipped")
    for name in skipped:
        print("  skipped", name)
    for (backend, structure, kind, stamp), items in sorted(groups.items()):
        rows = sorted((r["parameters"]["num_data_qubits"], r["parameters"]["depth"],
                       r["benchmark"]["r"], r["benchmark"]["r_ovl"]) for _, r in items)
        print(f"\n  {backend}/{structure}/{kind}/{run_filename(backend, structure, kind, stamp)}")
        for n, depth, r, ovl in rows:
            print(f"    nq={n:<3d} p={depth:<3d} r={r:.4f}  r_ovl={ovl:.4f}")
    if not args.apply:
        print("\nreport only - re-run with --apply to write")
        return

    for (backend, structure, kind, stamp), items in sorted(groups.items()):
        records = [r for _, r in items]
        run = {"backend": backend, "structure": structure, "kind": kind, "stamp": stamp,
               "simulated": False,
               "shots": sorted({r["parameters"]["shots"] for r in records}),
               "depths": sorted({r["parameters"]["depth"] for r in records}),
               "chain_lengths": sorted({r["parameters"]["num_data_qubits"] for r in records}),
               "source": "Benchmarking-Mid-circuit-measurement/Data",
               "source_files": sorted(p.name for p, _ in items)}
        out = save_run(Path(args.dest) / backend / structure / kind / run_filename(backend, structure, kind, stamp),
                       records, run=run)
        print("wrote", out)


if __name__ == "__main__":
    main()
