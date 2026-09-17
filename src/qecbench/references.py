"""Stored noiseless references ``<H>`` for LR-QAOA runs that are expensive to recompute.

An exact reference for 21-25 data qubits takes seconds to minutes. Beyond 25 there is no exact
reference, and a tensor-network estimate is a deliberate choice with its own error. Both are
stored here, keyed by the Hamiltonian, the depth and the ramp, together with how they were
obtained, so every notebook and analysis reads the same number.

The store is a JSON file, ``data/references/ideal_energies.json`` in this repository by default
(``QECBENCH_REFERENCES`` points it elsewhere).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def store_path() -> Path:
    default = Path(__file__).resolve().parents[2] / "data" / "references" / "ideal_energies.json"
    return Path(os.environ.get("QECBENCH_REFERENCES", default))


def key(hamiltonian, n: int, depth: int, delta: float) -> str:
    terms = sorted((sorted(int(q) for q in t), float(w)) for t, w in hamiltonian.items())
    blob = json.dumps({"n": int(n), "depth": int(depth), "delta": float(delta), "terms": terms})
    return hashlib.sha1(blob.encode()).hexdigest()


def _read() -> dict:
    path = store_path()
    return json.loads(path.read_text()) if path.exists() else {}


def lookup(hamiltonian, n: int, depth: int, delta: float):
    """The stored entry (``energy``, ``method``, ...) or None."""
    return _read().get(key(hamiltonian, n, depth, delta))


def store(hamiltonian, n: int, depth: int, delta: float, energy: float, method: str, **info):
    path = store_path()
    table = _read()
    table[key(hamiltonian, n, depth, delta)] = {
        "n": int(n), "depth": int(depth), "delta": float(delta), "n_terms": len(hamiltonian),
        "energy": float(energy), "method": method, "created": datetime.now().isoformat(timespec="seconds"),
        **info}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1, sort_keys=True))
    return table[key(hamiltonian, n, depth, delta)]
