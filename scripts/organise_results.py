"""Move older result files into the current layout, packing them into run files.

Earlier campaigns wrote one JSON per instance and depth into a single flat directory - a few
thousand files that are hard to search.  This reads them, groups them by device, problem,
implementation and the job they came from, and writes

    <destination>/<backend>/<structure>/<kind>/<stamp>_<backend>_<structure>_<kind>.json

with every result of a run inside one file.  Nothing is interpreted: each record is copied
across as it stands, so ``load_results`` reads the result exactly as before.

    python scripts/organise_results.py SOURCE [-d DEST] [--move] [--apply]

By default it only reports what it would do, and it copies rather than moves; pass ``--apply``
to write and ``--move`` to delete the originals afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qecbench.analysis import instance_from_record, records_in, run_filename, save_run  # noqa: E402


def stamp_of(record, path):
    """The run a record belongs to: its manifest if it names one, else its own timestamp."""
    manifest = (record.get("benchmark") or {}).get("manifest")
    if manifest and manifest[:8].isdigit():
        return "_".join(manifest.split("_")[:2])
    name = path.name
    if name[:8].isdigit():
        return "_".join(name.split("_")[:2])
    stamp = record.get("metadata", {}).get("timestamp")
    return datetime.fromisoformat(stamp).strftime("%Y%m%d_%H%M") if stamp else "unknown"


def collect(source):
    """``{(backend, structure, kind, stamp): [records]}`` for every result file under source."""
    groups, files, skipped = defaultdict(list), 0, 0
    for path in sorted(Path(source).rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            skipped += 1
            continue
        found = records_in(document)
        if not found:
            skipped += 1
            continue
        files += 1
        for record in found:
            instance = instance_from_record(record)
            backend = record.get("metadata", {}).get("backend")
            if instance is None or not backend:
                skipped += 1
                continue
            kind = record["parameters"].get("kind") or ("mcm" if instance.ancillas else "direct")
            groups[(backend, instance.structure, kind, stamp_of(record, path))].append((path, record))
    return groups, files, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="directory holding the existing result files")
    parser.add_argument("-d", "--dest", default="data/results", help="where the tree is written")
    parser.add_argument("--apply", action="store_true", help="actually write (default: report only)")
    parser.add_argument("--move", action="store_true", help="delete the originals once written")
    args = parser.parse_args()

    groups, files, skipped = collect(args.source)
    total = sum(len(v) for v in groups.values())
    print(f"{files} files read from {args.source}: {total} results, {skipped} skipped\n")
    print(f"{'destination':<62} {'results':>8}")
    for (backend, structure, kind, stamp) in sorted(groups):
        records = groups[(backend, structure, kind, stamp)]
        name = run_filename(backend, structure, kind, stamp)
        print(f"  {backend}/{structure}/{kind}/{name:<40} {len(records):>6}")
    print(f"\n{len(groups)} run file(s) would replace {files} file(s)"
          + ("" if args.apply else "   - re-run with --apply to write them"))
    if not args.apply:
        return

    written = 0
    for (backend, structure, kind, stamp), records in sorted(groups.items()):
        folder = Path(args.dest) / backend / structure / kind
        save_run(folder / run_filename(backend, structure, kind, stamp),
                 [r for _, r in records], run={"backend": backend, "migrated_from": str(args.source)})
        written += len(records)
        if args.move:
            for path in {p for p, _ in records}:
                path.unlink(missing_ok=True)
    print(f"wrote {written} results into {len(groups)} run file(s) under {args.dest}"
          + (f", removed {files} original file(s)" if args.move else ""))


if __name__ == "__main__":
    main()
