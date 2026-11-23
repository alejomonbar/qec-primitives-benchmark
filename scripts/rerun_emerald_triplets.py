"""Run the iqm_emerald MCM triplet survey again, on the same 104 triplets as the 13 August 2026 run.

The CLI picks one triplet per ancilla (``select_chains``); the 13 August run measured every triplet through 37
ancillas (up to 6 per ancilla), so this script takes the triplets from the stored run instead:
depths 3, 6, 9, 12, 500 shots, MCM only (68 circuits, about 75 USD on Braket).

    python scripts/rerun_emerald_triplets.py            # print the plan and its cost; nothing is sent
    python scripts/rerun_emerald_triplets.py --local    # rehearse on the IQM simulator
    python scripts/rerun_emerald_triplets.py --submit   # send it (AWS credentials, region eu-north-1)
    python -m qecbench collect                          # later: fetch the results into data/results/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qecbench.analysis import load_results  # noqa: E402
from qecbench.cli import _ledger_add, collect, load_config, make_backend  # noqa: E402
from qecbench.experiment import build_plan, print_plan, submit  # noqa: E402

RUNS = {"20260813_0722_iqm_emerald_chain_mcm.json", "20260813_0723_iqm_emerald_chain_mcm.json"}
DEPTHS, SHOTS = [3, 6, 9, 12], 500

local, send = "--local" in sys.argv, "--submit" in sys.argv or "--local" in sys.argv
data = ROOT / "data"
cfg = load_config(ROOT / "configs" / "iqm_chain.toml", nq=[2], ps=DEPTHS, shots=SHOTS, direct=False, local=local,
                  data_dir=str(data))
backend, cal = make_backend(cfg)
triplets = sorted(load_results(data / "results", "iqm_emerald", kind="mcm", n_data=2, files=RUNS))
plan = build_plan(backend, triplets, depths=DEPTHS, shots=SHOTS, delta=cfg["delta"])
print_plan(plan, backend)
if not send:
    print("\nplan only: add --submit to send it, or --local to rehearse")
    sys.exit()
manifest = submit(plan, backend, manifest_dir=data / "manifests")
_ledger_add(data, manifest, cfg, backend.name)
if local:
    collect([Path(manifest)], data)
else:
    print("\nsubmitted. Check with   python -m qecbench status\ncollect with             python -m qecbench collect")
