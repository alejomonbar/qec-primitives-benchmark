"""Bring the inputs of the two-device shot budget (paper Fig. 8a) into this repository.

The question: two devices whose error rates differ by a factor of two - how many shots per device does
LR-QAOA need to rank them at ``z`` sigma, ``S* = z^2 (sigma_A^2 + sigma_B^2) / (mu_B - mu_A)^2``, where ``mu``
and ``sigma`` are the mean and spread of the single-shot ``r``?

* Surface code ``d = 3``: ``mu`` and ``sigma`` of noisy simulations, 50 000 shots per device and depth, of
  the direct circuit with a two-qubit depolarizing channel of strength ``lambda`` after every CNOT
  (``scripts/two_device_gap_surface.py`` of the MCM repository). Only these moments were kept there, and
  they are all ``S*`` needs, so they are copied.
* Colour code ``d = 5``: a white-noise model, ``P = A P_ideal + (1 - A) P_random`` with
  ``A = 2^(-c p lambda)`` and ``c = kappa_0(n) N_CX`` (``scripts/two_device_ranking_color.py``). Only the
  model constant ``c`` and the depths are copied; the distributions are exact and recomputed
  (``qecbench.shots``).

    python scripts/import_shot_budget.py [SOURCE_DIR] [--apply]

writes ``data/shot_budget/two_device_ranking.json``. It only reports unless ``--apply`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "shot_budget" / "two_device_ranking.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=str(ROOT.parent / "Benchmarking-Mid-circuit-measurement" / "Data" / "figure_data"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    src = Path(args.source)

    surface = {}
    for name in ("two_device_gap_surface_d3.json", "two_device_gap_surface_d3_p002_p001.json"):
        d = json.loads((src / name).read_text())
        lam = d["meta"]["lambdas"]
        surface[f"{lam['worse']:g}/{lam['better']:g}"] = {
            "lambda_worse": lam["worse"], "lambda_better": lam["better"], "shots_per_depth": d["meta"]["n_ref"],
            "source_file": name,
            "moments": {p: {"mean_worse": v["muA"], "std_worse": v["sigA"], "mean_better": v["muB"], "std_better": v["sigB"]}
                        for p, v in d["per_depth"].items()}}
    color = json.loads((src / "two_device_ranking_color.json").read_text())
    model = {f"{v['lam_worse']:g}/{v['lam_better']:g}": {"lambda_worse": v["lam_worse"], "lambda_better": v["lam_better"],
                                                        "depths": v["depths"]}
             for k, v in color.items() if k.startswith("color_d5")}
    c = next(v["c"] for k, v in color.items() if k.startswith("color_d5"))
    out = {
        "question": "shots per device to rank two devices whose error rates differ by a factor of two: "
                    "S* = z^2 (std_worse^2 + std_better^2) / (mean_better - mean_worse)^2, single-shot r",
        "surface_d3": {"code": "surface_d3", "kind": "direct", "p_layers_delta": 0.5,
                       "noise": "two-qubit depolarizing channel of strength lambda after every CNOT (Aer)",
                       "source": "Benchmarking-Mid-circuit-measurement/scripts/two_device_gap_surface.py",
                       "pairs": surface},
        "color_d5_model": {"code": "color_d5", "kind": "direct",
                           "model": "P = A P_ideal + (1 - A) P_random, A = 2^(-c p lambda), c = kappa_0(n) N_CX",
                           "c": c, "n_cx_per_layer": 66,
                           "source": "Benchmarking-Mid-circuit-measurement/scripts/two_device_ranking_color.py "
                                     "(kappa_0 from Data/k_saturation_fit_color_code.json)",
                           "pairs": model},
    }
    print(json.dumps(out, indent=1)[:2000])
    if args.apply:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1))
        print("wrote", OUT)
    else:
        print("\nreport only - re-run with --apply to write")


if __name__ == "__main__":
    main()
