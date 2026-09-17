"""Re-harvest an earlier IBM surface-code position scan into this package's format.

A position scan runs, on each of several surface-code patches of a
square-lattice chip, the LR-QAOA benchmark at depths ``p`` and a surface-code memory experiment at ``R`` rounds in
the Z and X bases, all on the same physical qubits. Its manifest names the IBM jobs and every task; the
earlier analysis kept only aggregated numbers. This fetches the finished jobs again (read-only, no QPU time) and writes

    <dest>/<backend>/surface_code/mcm/<stamp>_<backend>_surface_code_mcm.json        LR-QAOA, one record per patch/depth
    <dest>/<backend>/surface_code/memory/<stamp>_<backend>_surface_code_memory.json  memory, raw shots packed

Each patch is rebuilt from its anchor ``(r0, c0)`` with ``layout.surface_code_placements`` (which reproduces the
scan's layout qubit for qubit). The calibration snapshot the manifest stored for each patch goes into the run header,
``run.calibration["d<d>_r<r0>_c<c0>"]``, and every result names its entry.

    python scripts/import_position_scan.py MANIFEST [--account NAME] [--grid 12x10] [-d DEST] [--apply]

It only reports unless ``--apply`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qecbench import codes, memory  # noqa: E402
from qecbench.analysis import make_record, run_filename, save_run  # noqa: E402
from qecbench.layout import surface_code_placements  # noqa: E402


def calibration_key(key):
    """``d3_r7_c2``: how a patch's calibration snapshot is named in the run header."""
    return f"d{key[0]}_r{key[1]}_c{key[2]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--account", default="mcm-primitives")
    parser.add_argument("--grid", default="12x10", help="rows x cols of the square-lattice chip")
    parser.add_argument("-d", "--dest", default="data/results")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    backend = manifest["backend"]
    rows, cols = (int(v) for v in args.grid.split("x"))
    G = nx.relabel_nodes(nx.grid_2d_graph(rows, cols), lambda rc: rc[0] * cols + rc[1])
    patches, calibration = {}, {}
    for d, positions in manifest["positions"].items():
        code = codes.surface_code(int(d))
        for pos in positions:
            (patch,) = surface_code_placements(G, code, anchors=[(pos["r0"], pos["c0"])])
            if sorted(pos["cal"]["data_qubits"]) != sorted(patch.data_qubits):
                raise ValueError(f"anchor {(pos['r0'], pos['c0'])}: data qubits differ from the manifest's")
            patches[(int(d), pos["r0"], pos["c0"])] = patch
            calibration[(int(d), pos["r0"], pos["c0"])] = pos["cal"]

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(name=args.account)
    stamp = datetime.fromisoformat(manifest["created"]).strftime("%Y%m%d_%H%M")
    source = {"source": "legacy position scan", "manifest": manifest_path.name}
    memory_meta = {"backend": backend, "simulated": False, "noise_model": None, "p_2q_model": manifest["p_2q_model"]}
    lrqaoa, memories = [], []
    for job in manifest["jobs"]:
        result = service.job(job["job_id"]).result()
        for task in job["tasks"]:
            key = (task["d"], task["r0"], task["c0"])
            patch, pub = patches[key], result[task["index"]]
            extra = {**source, "anchor": [task["r0"], task["c0"]], "calibration": f"run.calibration[{calibration_key(key)}]"}
            if task["arm"] == "lrqaoa":
                counts = {}
                for bits, n in pub.data.c.get_counts().items():   # register c, bit 0 rightmost
                    counts[bits[::-1][:patch.n_data]] = counts.get(bits[::-1][:patch.n_data], 0) + int(n)
                rec = make_record(counts, patch, depth=task["depth"], delta=manifest["delta"], backend_name=backend,
                                  job_id=job["job_id"], kind="mcm",
                                  extra={**extra, "program": "square-lattice MCM LR-QAOA (legacy builder)",
                                         "circuit_variant": "square-lattice embedding, the CZ of every check in 4 rounds"})
                rec["energy_analysis"].pop("energy_values", None)
                lrqaoa.append(rec)
                print(f"  lrqaoa d={task['d']} ({task['r0']},{task['c0']}) p={task['depth']:<3} "
                      f"r={rec['benchmark']['r']:.4f} r_ovl={rec['benchmark']['r_ovl']:.4f}")
            else:
                counts = memory._device_counts(pub, task["rounds"])
                rec = memory.memory_record(counts, patch, task["rounds"], task.get("basis", "Z"), memory_meta,
                                           job["job_id"], extra={**extra, "program": "surface-code memory (legacy builder, the schedule of qecbench.memory)"})
                memories.append(rec)
                b = rec["benchmark"]
                print(f"  memory d={task['d']} ({task['r0']},{task['c0']}) {rec['parameters']['basis']} "
                      f"R={task['rounds']:<3} P_L={b['logical_error_rate']:.4f}+-{b['logical_error_rate_err']:.4f}")
    print(f"{len(lrqaoa)} LR-QAOA and {len(memories)} memory results")
    if not args.apply:
        print("report only - re-run with --apply to write")
        return
    header = {"backend": backend, "created": manifest["created"], "manifest": manifest_path.name, "simulated": False,
              **source, "calibration": {calibration_key(k): v for k, v in calibration.items()}}
    out = save_run(Path(args.dest) / backend / "surface_code" / "mcm" / run_filename(backend, "surface_code", "mcm", stamp),
                   lrqaoa, run={**header, "depths": manifest["lrqaoa_depths"], "delta": manifest["delta"]})
    print("wrote", out)
    out = save_run(Path(args.dest) / backend / "surface_code" / "memory" / f"{stamp}_{backend}_surface_code_memory.json",
                   memories, run={**header, "experiment": "memory", "rounds": manifest["qec_rounds"],
                                  "bases": manifest.get("qec_bases", ["Z"]), "p_2q_model": manifest["p_2q_model"]})
    print("wrote", out)


if __name__ == "__main__":
    main()
