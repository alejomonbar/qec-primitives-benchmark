"""QEC-code structures as LR-QAOA Hamiltonians, and how to schedule their checks.

A code structure is a set of **checks**: each is a support of data qubits and becomes one
term ``w * Z...Z`` of an Ising Hamiltonian

    H = sum_c w_c prod_{i in c} Z_i .

Measured through an ancilla - CNOTs from the support, a phase rotation, a mid-circuit
measurement and feed-forward - each term is exactly one syndrome-extraction gadget, so an
LR-QAOA layer is one round of syndrome extraction and depth ``p`` is ``p`` rounds. How much of
the noiseless answer survives measures how a device copes with the code's own pattern of
checks, before any encoding or decoding (the "QEC structure evaluation").

Nothing downstream depends on which code it is: a structure is only ``n_data`` and its
weighted checks, so a surface code, a colour code, a qLDPC code or any Hamiltonian you load
all run through the same scheduler, circuit builders, backends and analysis.

Generators
----------
``surface_code(d)``          the rotated-surface-code-like Hamiltonian of the reference
                             study: ``(d-1)^2`` weight-4 bulk checks and ``2(d-1)`` weight-2
                             boundary checks on a ``d x d`` grid of data qubits.
``color_code(d)``            the triangular 6.6.6 colour code, weight-6 bulk and weight-4
                             boundary faces (odd ``d >= 3``).
``bivariate_bicycle(name)``  quasi-cyclic qLDPC codes (``BB18``, ``BB24``, ``BB30``, ``BB48``,
                             ``GB16``, ``GB26``): the independent rows of ``H_X = [A|B]`` and
                             ``H_Z = [B^T|A^T]``, merged where supports coincide (weight 2).
``from_checks``              any structure, from a list of supports (duplicates become weights).
``from_hamiltonian``         any structure, from ``{(i, j, ...): weight}``.
``load``                     any structure, from a file (see ``load`` for the forms it reads).

Schedules
---------
``schedule(structure, max_parallel)`` partitions the checks into **batches** of disjoint
supports: the checks of a batch touch no common data qubit, so their gadgets are independent
and can run at once. Fewer batches per round means less idling. ``max_parallel`` caps a batch,
for a device that can only hold or operate on so many ancillas at a time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

FAMILIES = ("surface_code", "color_code", "qldpc", "custom")


@dataclass(frozen=True)
class CodeStructure:
    """The checks of a code, as an Ising Hamiltonian on data qubits ``0..n_data-1``."""

    name: str
    family: str
    n_data: int
    checks: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...] = ()
    info: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        checks = tuple(tuple(int(q) for q in c) for c in self.checks)
        object.__setattr__(self, "checks", checks)
        weights = tuple(float(w) for w in self.weights) if self.weights else (1.0,) * len(checks)
        object.__setattr__(self, "weights", weights)
        if len(weights) != len(checks):
            raise ValueError(f"{self.name}: {len(weights)} weights for {len(checks)} checks")
        for c in checks:
            if len(c) < 1 or len(set(c)) != len(c) or min(c) < 0 or max(c) >= self.n_data:
                raise ValueError(f"{self.name}: invalid check {c} for {self.n_data} data qubits")
        if len(set(checks)) != len(checks):
            raise ValueError(f"{self.name}: repeated check; merge duplicates into a weight instead")

    # -- the Hamiltonian ----------------------------------------------------------------
    @property
    def hamiltonian(self) -> dict[tuple[int, ...], float]:
        return dict(zip(self.checks, self.weights))

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def check_weights(self) -> dict[int, int]:
        """``{support size: number of checks}``."""
        out = {}
        for c in self.checks:
            out[len(c)] = out.get(len(c), 0) + 1
        return dict(sorted(out.items()))

    def counts(self, kind: str, depth: int = 1) -> dict:
        """Two-qubit gates and mid-circuit measurements of ``depth`` layers.

        ``mcm``: one CNOT per support qubit and one measurement per check. ``direct``: a CNOT
        ladder onto the last support qubit, ``2(w - 1)`` CNOTs, no measurement. A weight-1
        check needs no two-qubit gate either way. Taken from the checks themselves, so they
        hold for any structure.
        """
        if kind == "mcm":
            n2q = sum(len(c) for c in self.checks)
            nm = self.n_checks
        elif kind == "direct":
            n2q, nm = sum(2 * (len(c) - 1) for c in self.checks), 0
        else:
            raise ValueError(f"unknown kind {kind!r}")
        return {"two_qubit_gates": n2q * depth, "mid_circuit_measurements": nm * depth}

    # -- reference energies ---------------------------------------------------------------
    def max_energy(self) -> float:
        """All-zero bitstring: every check reads +1, so ``E = sum w`` (weights are positive)."""
        if min(self.weights) < 0:
            return float(_extreme_energy(self, maximise=True))
        return float(sum(self.weights))

    @cached_property
    def _optimum(self) -> float:
        return float(_extreme_energy(self, maximise=False))

    def optimal_energy(self) -> float:
        """Exact ground-state energy (see ``_extreme_energy``)."""
        return self._optimum

    # -- serialisation ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "n_data": self.n_data,
                "checks": [list(c) for c in self.checks], "weights": list(self.weights),
                "info": self.info}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["family"], int(d["n_data"]), tuple(tuple(c) for c in d["checks"]),
                   tuple(d.get("weights") or ()), dict(d.get("info") or {}))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    def __repr__(self):
        return (f"CodeStructure({self.name!r}, {self.family}, n_data={self.n_data}, "
                f"checks={self.check_weights})")


def _extreme_energy(structure: CodeStructure, maximise: bool) -> float:
    """Exact extreme energy of ``sum w_c Z_c`` as a mixed-integer programme.

    With ``z_i = 1 - 2 x_i``, a check reads ``-1`` when the parity of its support is odd:
    ``sum_{i in c} x_i = 2 k_c + y_c`` with integer ``k_c`` and binary ``y_c``. The energy is
    ``sum_c w_c (1 - 2 y_c)``. Solved exactly by HiGHS (``scipy.optimize.milp``), which handles
    the codes here (up to ~100 data qubits) in well under a second. Every check being odd at
    once is a linear system over GF(2); when it is solvable the answer is ``-sum w`` directly.
    """
    n, checks, weights = structure.n_data, structure.checks, np.asarray(structure.weights)
    if not maximise and min(weights) > 0 and _all_odd_solvable(n, checks):
        return -float(weights.sum())
    from scipy.optimize import Bounds, LinearConstraint, milp

    m = len(checks)
    # variables: x (n binary), y (m binary), k (m integer)
    n_var = n + 2 * m
    cost = np.zeros(n_var)
    cost[n:n + m] = 2 * weights if maximise else -2 * weights      # minimise -(energy) or energy
    A = np.zeros((m, n_var))
    for c, support in enumerate(checks):
        A[c, list(support)] = 1
        A[c, n + c] = -1
        A[c, n + m + c] = -2
    upper_k = np.array([len(s) // 2 for s in checks], dtype=float)
    bounds = Bounds(np.zeros(n_var), np.concatenate([np.ones(n + m), upper_k]))
    res = milp(cost, constraints=LinearConstraint(A, 0, 0), integrality=np.ones(n_var), bounds=bounds)
    if not res.success:
        raise RuntimeError(f"{structure.name}: ground-state search failed ({res.message})")
    y = np.round(res.x[n:n + m])
    return float((weights * (1 - 2 * y)).sum())


def _all_odd_solvable(n, checks) -> bool:
    """Whether ``H x = 1`` has a solution over GF(2): some bitstring makes every check odd."""
    rows = []
    for support in checks:
        v = 0
        for q in support:
            v |= 1 << q
        rows.append((v, 1))
    basis = {}
    for v, rhs in rows:
        while v:
            pivot = v.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (v, rhs)
                break
            bv, brhs = basis[pivot]
            v, rhs = v ^ bv, rhs ^ brhs
        else:
            if rhs:
                return False
    return True


# ======================================================================================
# Generators
# ======================================================================================
def surface_code(d: int) -> CodeStructure:
    """The surface-code-like Hamiltonian of the reference study on a ``d x d`` data grid.

    Data qubit ``i * d + j`` sits at row ``i``, column ``j``. Every unit square gives a weight-4
    check, and alternating boundary edges give ``2(d - 1)`` weight-2 checks, as in the rotated
    surface code (``utils.sc_hamiltonian`` of the reference study; the term order is the same).
    """
    if d < 2:
        raise ValueError("surface code distance must be >= 2")
    checks = []
    for j in range(d - 1):
        for i in range(d - 1):
            row_1, row_2 = j * d, (j + 1) * d
            checks.append((i + row_1, i + 1 + row_1, i + row_2, i + 1 + row_2))
    for i in range(d - 1):
        if i % 2 == 0:
            col_0 = i * d
            checks.append((col_0, col_0 + d))
            row_d = (d - 1) * d
            checks.append((i + row_d, i + 1 + row_d))
        else:
            checks.append((i, i + 1))
            col_d = i * d + (d - 1)
            checks.append((col_d, col_d + d))
    positions = {i * d + j: (j, d - 1 - i) for i in range(d) for j in range(d)}
    return CodeStructure(f"surface_d{d}", "surface_code", d * d, tuple(checks),
                         info={"distance": d, "positions": positions})


def color_code(d: int) -> CodeStructure:
    """The triangular 6.6.6 colour code of odd distance ``d``: its faces as checks.

    Bulk faces have weight 6 and boundary faces weight 4; ``d = 11`` has 91 data qubits and
    45 checks. Built as in ``utils.color_code_graph`` of the reference study: the honeycomb
    rows are laid out explicitly, the faces read off a planar embedding, and the outer face
    dropped.
    """
    import networkx as nx

    if d < 3 or d % 2 == 0:
        raise ValueError("colour code distance must be odd and >= 3")
    M = (d - 1) // 2
    pos, node = {}, 0
    for row in range(3 * M + 1):
        y, m, kind = -row * math.sqrt(3), row // 3, row % 3
        if kind == 0:
            x, xs = -3 * m, [-3 * m]
            for i in range(2 * m):
                x += 2 if i % 2 == 0 else 4
                xs.append(x)
        elif kind == 1:
            x, xs = -(3 * m + 1), [-(3 * m + 1)]
            for i in range(2 * m):
                x += 4 if i % 2 == 0 else 2
                xs.append(x)
        else:
            x, xs = -3 * m, [-3 * m]
            for i in range(2 * m + 1):
                x += 2 if i % 2 == 0 else 4
                xs.append(x)
        for xi in xs:
            pos[node] = (float(xi), float(y))
            node += 1

    nodes = sorted(pos)
    coords = np.array([pos[q] for q in nodes])
    edges = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(np.linalg.norm(coords[i] - coords[j]) - 2.0) < 1e-4:
                edges.add((nodes[i], nodes[j]))
    edges |= _hull_edges(nodes, coords)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(sorted(edges))
    planar, embedding = nx.check_planarity(G)
    if not planar:
        raise RuntimeError("colour-code lattice is not planar")
    faces, seen = [], set()
    for v in embedding:
        for w in embedding.neighbors_cw_order(v):
            if (v, w) in seen:
                continue
            face, cv, cw = [], v, w
            while (cv, cw) not in seen:
                seen.add((cv, cw))
                face.append(cv)
                nxt = embedding.next_face_half_edge(cv, cw)
                cv, cw = cw, nxt[1]
            if len(face) >= 3:
                faces.append(face)
    faces.sort(key=len)
    faces = faces[:-1]                                   # drop the outer face
    return CodeStructure(f"color_d{d}", "color_code", len(nodes), tuple(tuple(f) for f in faces),
                         info={"distance": d, "positions": pos})


def _hull_edges(nodes, coords, tol=1e-4):
    """Edges between consecutive lattice points along the convex hull (the triangle's sides)."""
    from scipy.spatial import ConvexHull

    edges = set()
    for simplex in ConvexHull(coords).simplices:
        p1, p2 = coords[simplex[0]], coords[simplex[1]]
        direction = p2 - p1
        length = float(direction @ direction)
        on_side = []
        for idx, point in enumerate(coords):
            t = float((point - p1) @ direction) / length if length > 0 else 0.0
            if -tol <= t <= 1 + tol and np.linalg.norm(point - (p1 + np.clip(t, 0, 1) * direction)) < tol:
                on_side.append((t, nodes[idx]))
        on_side.sort()
        edges |= {tuple(sorted((a[1], b[1]))) for a, b in zip(on_side, on_side[1:])}
    return edges


QLDPC_CODES = {   # arXiv:2606.06455 App. A; BB30 and BB48 follow arXiv:2503.22071 Table 2
    "BB18": dict(label="BB[[18,4,3]]", ell=3, m=3, k=4, distance=3, A=[(0, 0), (1, 0)], B=[(0, 0), (0, 1), (1, 2)]),
    "BB24": dict(label="BB[[24,4,4]]", ell=4, m=3, k=4, distance=4, A=[(0, 0), (1, 0)], B=[(0, 0), (0, 1), (1, 2)]),
    "BB30": dict(label="BB[[30,4,5]]", ell=5, m=3, k=4, distance=5, A=[(0, 0), (1, 0)], B=[(0, 0), (0, 1), (2, 2)]),
    "BB48": dict(label="BB[[48,4,7]]", ell=8, m=3, k=4, distance=7, A=[(0, 0), (1, 0)], B=[(0, 0), (0, 1), (3, 2)]),
    "GB16": dict(label="GB[[16,2,4]]", ell=1, m=8, k=2, distance=4, A=[(0, 0), (0, 1)], B=[(0, 0), (0, 5)]),
    "GB26": dict(label="GB[[26,2,5]]", ell=1, m=13, k=2, distance=5, A=[(0, 0), (0, 10)], B=[(0, 9), (0, 11)]),
}


def bivariate_bicycle(name: str) -> CodeStructure:
    """A quasi-cyclic qLDPC code from ``QLDPC_CODES``: its independent X and Z checks.

    ``A`` and ``B`` are sums of shift monomials ``x^a y^b`` on the ``ell x m`` torus. The checks
    are the GF(2)-independent rows of ``H_X = [A | B]`` and ``H_Z = [B^T | A^T]`` (first rows
    that raise the rank, in order), ``n - k`` in all. An X and a Z check on the same support
    become one term of weight 2.
    """
    key = name.upper().replace(" ", "")
    spec = QLDPC_CODES.get(key) or next((s for s in QLDPC_CODES.values() if s["label"] == key), None)
    if spec is None:
        raise ValueError(f"unknown qLDPC code {name!r}; have {list(QLDPC_CODES)}")
    key = next(k for k, s in QLDPC_CODES.items() if s is spec)
    ell, m = spec["ell"], spec["m"]
    A, B = _quasi_cyclic(ell, m, spec["A"]), _quasi_cyclic(ell, m, spec["B"])
    Hx, Hz = np.concatenate([A, B], axis=1), np.concatenate([B.T, A.T], axis=1)
    x_rows, z_rows = _independent_rows(Hx), _independent_rows(Hz)
    n = 2 * ell * m
    if len(x_rows) + len(z_rows) != n - spec["k"]:
        raise ValueError(f"{key}: found {len(x_rows) + len(z_rows)} independent checks, expected {n - spec['k']}")
    merged = {}
    for H, rows in ((Hx, x_rows), (Hz, z_rows)):
        for r in rows:
            support = tuple(np.flatnonzero(H[r]).tolist())
            merged[support] = merged.get(support, 0.0) + 1.0
    return CodeStructure(key, "qldpc", n, tuple(merged), tuple(merged.values()),
                         info={**{k: v for k, v in spec.items() if k not in ("A", "B")},
                               "A_terms": spec["A"], "B_terms": spec["B"],
                               "x_checks": len(x_rows), "z_checks": len(z_rows)})


def _circulant(size, shift):
    matrix = np.zeros((size, size), dtype=np.uint8)
    cols = np.arange(size)
    matrix[(cols + shift) % size, cols] = 1
    return matrix


def _quasi_cyclic(ell, m, terms):
    matrix = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for a, b in terms:
        matrix ^= np.kron(_circulant(ell, a % ell), _circulant(m, b % m)).astype(np.uint8)
    return matrix


def _independent_rows(H):
    basis, selected = {}, []
    for index, row in enumerate(np.asarray(H, dtype=np.uint8) % 2):
        v = int("".join(str(b) for b in row[::-1]), 2) if row.any() else 0
        while v:
            pivot = v.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = v
                selected.append(index)
                break
            v ^= basis[pivot]
    return selected


def from_checks(name: str, n_data: int, checks, weights=None, family: str = "custom", **info) -> CodeStructure:
    """Any structure: ``checks`` are supports on ``0..n_data-1``; duplicates merge into weights."""
    merged = {}
    for i, c in enumerate(checks):
        key = tuple(int(q) for q in c)
        merged[key] = merged.get(key, 0.0) + (1.0 if weights is None else float(weights[i]))
    return CodeStructure(name, family, int(n_data), tuple(merged), tuple(merged.values()), info=info)


def from_hamiltonian(name: str, hamiltonian, n_data: int | None = None, family: str = "custom",
                     **info) -> CodeStructure:
    """A structure from an Ising Hamiltonian ``{(i, j, ...): weight}``.

    Every term is a check, whatever its weight, so any code whose checks are Z-products (or any
    other diagonal Hamiltonian without a constant) runs as it is. ``n_data`` defaults to one more
    than the largest qubit index.
    """
    terms = {tuple(int(q) for q in t): float(w) for t, w in dict(hamiltonian).items()}
    if n_data is None:
        n_data = 1 + max(q for t in terms for q in t)
    return from_checks(name, n_data, list(terms), list(terms.values()), family=family, **info)


def load(path, name: str | None = None, family: str = "custom") -> CodeStructure:
    """A structure from a file, whichever of these forms it has:

    * ``.json`` written by ``CodeStructure.save`` (name and family come from the file);
    * ``.json`` result file of an earlier campaign, read from its ``hamiltonian`` block
      (``hamiltonian_couplings``, ``hamiltonian_weights``, ``num_nodes``);
    * ``.txt`` with a ``#n_qubits:<n>`` header and one check per line as a list, ``[0, 6, 9]``,
      optionally followed by its weight (the ``graphs/*_hamiltonian.txt`` files of the reference
      study).

    ``name`` defaults to the file name without ``_hamiltonian`` and the extension.
    """
    path = Path(path)
    default = name or path.stem.removesuffix("_hamiltonian")
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        if "checks" in d:
            structure = CodeStructure.from_dict(d)
            return structure if name is None else CodeStructure.from_dict({**d, "name": name})
        ham = d.get("hamiltonian", d)
        couplings = ham["hamiltonian_couplings"]
        weights = ham.get("hamiltonian_weights") or [1.0] * len(couplings)
        return from_hamiltonian(default, dict(zip(map(tuple, couplings), weights)), ham.get("num_nodes"), family,
                                source=path.name)
    n_data, terms = None, {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            if key.strip() in ("n_qubits", "n_data", "num_nodes"):
                n_data = int(value)
            continue
        support, _, rest = line.partition("]")
        check = tuple(int(q) for q in support.strip("[ ").replace(",", " ").split())
        terms[check] = terms.get(check, 0.0) + (float(rest.strip(" ,:")) if rest.strip(" ,:") else 1.0)
    return from_hamiltonian(default, terms, n_data, family, source=path.name)


def build(family: str, size) -> CodeStructure:
    """``build("surface_code", 3)``, ``build("color_code", 5)``, ``build("qldpc", "BB18")``."""
    if family == "surface_code":
        return surface_code(int(size))
    if family == "color_code":
        return color_code(int(size))
    if family == "qldpc":
        return bivariate_bicycle(str(size))
    raise ValueError(f"unknown code family {family!r}; have surface_code, color_code, qldpc")


# ======================================================================================
# Schedules
# ======================================================================================
@dataclass(frozen=True)
class Schedule:
    """Batches of check indices; within a batch the supports are disjoint."""

    structure: CodeStructure
    batches: tuple[tuple[int, ...], ...]

    @property
    def n_batches(self) -> int:
        return len(self.batches)

    @property
    def max_batch(self) -> int:
        return max(map(len, self.batches))

    def idle_qubit_steps(self) -> int:
        """``sum over batches of (n_data - |union of supports|)`` per round (reference study)."""
        n = self.structure.n_data
        return sum(n - len({q for c in batch for q in self.structure.checks[c]}) for batch in self.batches)

    def peak_qubits(self, kind: str = "mcm") -> int:
        """Data qubits plus the ancillas one batch holds at once (``kind='mcm'``)."""
        return self.structure.n_data + (self.max_batch if kind == "mcm" else 0)

    def summary(self) -> dict:
        return {"batches_per_round": self.n_batches, "largest_batch": self.max_batch,
                "idle_qubit_steps_per_round": self.idle_qubit_steps()}


def schedule(structure: CodeStructure, max_parallel: int | None = None, strategy: str = "DSATUR") -> Schedule:
    """Partition the checks into batches of disjoint supports.

    The checks that share a data qubit conflict. A greedy colouring of that conflict graph
    (``strategy`` as in ``networkx.greedy_color``; ``DSATUR`` gives the fewest colours for these
    lattices, e.g. 4 for the surface code) gives classes that can run in parallel. Each class
    is then cut into chunks of at most ``max_parallel`` checks, largest-support checks first.
    """
    import itertools

    import networkx as nx

    checks = structure.checks
    conflict = nx.Graph()
    conflict.add_nodes_from(range(len(checks)))
    by_qubit = {}
    for i, c in enumerate(checks):
        for q in c:
            by_qubit.setdefault(q, []).append(i)
    for members in by_qubit.values():
        conflict.add_edges_from(itertools.combinations(members, 2))
    colouring = nx.coloring.greedy_color(conflict, strategy=strategy)
    classes = {}
    for check, colour in colouring.items():
        classes.setdefault(colour, []).append(check)
    batches = []
    for colour in sorted(classes, key=lambda c: (-len(classes[c]), c)):
        members = sorted(classes[colour], key=lambda i: (-len(checks[i]), i))
        size = len(members) if not max_parallel else max_parallel
        batches += [tuple(members[k:k + size]) for k in range(0, len(members), size)]
    out = Schedule(structure, tuple(batches))
    for batch in out.batches:
        support = [q for c in batch for q in checks[c]]
        assert len(support) == len(set(support)), "batch with overlapping supports"
    return out
