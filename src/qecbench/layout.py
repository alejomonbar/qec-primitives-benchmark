"""Where primitives go on a chip: coupling graphs, instance selection, packing and checks.

Vendor-neutral.  Ported from ``iqm_mcm_auto`` / ``ibm_mcm_auto`` and generalised from
triplets to chains of any length:

* ``select_chains`` benchmarks every ancilla-capable qubit (at least) once.  For triplets
  it is exactly ``unique_ancilla_triples``: one neighbour pair per ancilla, chosen to level
  the qubit load, because ``max(load)`` is a hard lower bound on the number of circuits.
  For longer chains it is a greedy set cover of the candidate ancillas by paths.
* ``pack`` colours the conflict graph so qubit-disjoint instances share a circuit.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

import networkx as nx

from .primitives import Chain, Direct

DEVICE_DIR = Path(__file__).parent / "devices"


# --------------------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------------------
def graph_from_edges(edges) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from((int(u), int(v)) for u, v in edges)
    return G


def load_cached_graph(name: str, directory=DEVICE_DIR):
    path = Path(directory) / f"{name}_coupling.json"
    if not path.exists():
        return None
    return graph_from_edges(json.loads(path.read_text())["edges"])


def save_graph(name: str, G: nx.Graph, directory=DEVICE_DIR):
    path = Path(directory) / f"{name}_coupling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = sorted(tuple(sorted(e)) for e in G.edges)
    path.write_text(json.dumps({"qpu": name, "edges": [list(e) for e in edges]}, indent=1))
    return path


def synthetic_graph(topology: str = "heavy_hex", qubits: int = 54) -> nx.Graph:
    """A vendor-free coupling map for simulation: ``heavy_hex``, ``grid`` or ``line``.

    ``heavy_hex`` puts a qubit in the middle of every coupler of a hexagonal lattice, which is
    the shape IBM's chips have (degree <= 3, plenty of degree-2 qubits).  The lattice is grown
    until it is big enough and then cut back to exactly ``qubits`` nodes along a breadth-first
    walk, so the result stays connected.
    """
    if topology == "line":
        G = nx.path_graph(qubits)
    elif topology == "grid":
        side = int(qubits ** 0.5) + 1
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    elif topology == "heavy_hex":
        rows = 1
        while rows < 12:
            base = nx.hexagonal_lattice_graph(rows, rows)
            if base.number_of_nodes() + base.number_of_edges() >= qubits:
                break
            rows += 1
        G = nx.Graph()
        for u, v in nx.hexagonal_lattice_graph(rows, rows).edges:
            G.add_edge(u, ("mid", u, v))
            G.add_edge(("mid", u, v), v)
        # lattice nodes are (i, j) and the new ones ("mid", u, v): not mutually comparable, so
        # relabel in insertion order rather than sorting them
        G = nx.convert_node_labels_to_integers(G)
    else:
        raise ValueError(f"unknown topology {topology!r}: use heavy_hex, grid or line")
    if G.number_of_nodes() > qubits:
        keep = list(nx.bfs_tree(G, min(G.nodes)))[:qubits]   # a BFS prefix stays connected
        G = nx.convert_node_labels_to_integers(G.subgraph(keep).copy(), ordering="sorted")
    return G


def grid_layout(G: nx.Graph) -> dict:
    """(x, y) positions for chips numbered row by row (IQM square lattice, IBM heavy-hex).

    Consecutive labels are horizontal bonds, anything else is vertical.  Falls back to a
    Kamada-Kawai layout when the walk does not produce a clean unit-bond grid.
    """
    if G.number_of_nodes() == 0:
        return {}
    pos, stack = {min(G.nodes): (0, 0)}, [min(G.nodes)]
    while stack:
        u = stack.pop()
        x, y = pos[u]
        for v in G[u]:
            if v in pos:
                continue
            if abs(v - u) == 1:
                pos[v] = (x + 1, y) if v > u else (x - 1, y)
            else:
                pos[v] = (x, y - 1) if v > u else (x, y + 1)
            stack.append(v)
    clean = (len(pos) == G.number_of_nodes() and len(set(pos.values())) == len(pos)
             and all(abs(pos[u][0] - pos[v][0]) + abs(pos[u][1] - pos[v][1]) == 1
                     for u, v in G.edges))
    return pos if clean else nx.kamada_kawai_layout(G)


def restrict_to_groups(G: nx.Graph, groups) -> nx.Graph:
    """Drop every coupler between two feed-forward groups (IQM: control stays in a group)."""
    if not groups:
        return G
    grp = {q: g for g, qubits in groups.items() for q in qubits}
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from((u, v) for u, v in G.edges if grp.get(u) is not None and grp.get(u) == grp.get(v))
    return H


def region(G: nx.Graph, allowed_qubits=None) -> nx.Graph:
    return G if allowed_qubits is None else G.subgraph(allowed_qubits).copy()


# --------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------
def ancilla_candidates(G, allowed_qubits=None, ancillas=None):
    """Every qubit that can carry a mid-circuit measurement: degree >= 2 in the region."""
    H = region(G, allowed_qubits)
    pool = set(H.nodes) if ancillas is None else set(ancillas) & set(H.nodes)
    return sorted(a for a in pool if H.degree(a) >= 2)


def enumerate_chains(G, n_data=2, allowed_qubits=None, ancillas=None, limit=500_000):
    """Every chain of ``n_data`` data qubits on the coupling graph (``n_data=2``: all triples)."""
    H = region(G, allowed_qubits)
    length = 2 * n_data - 1
    allowed_anc = None if ancillas is None else set(ancillas)
    found = set()

    def extend(path):
        if len(found) > limit:
            raise RuntimeError(f"more than {limit} chains; restrict the region")
        if len(path) == length:
            if allowed_anc is None or set(path[1::2]) <= allowed_anc:
                found.add(Chain(path))
            return
        for v in H[path[-1]]:
            if v not in path:
                extend(path + [v])

    for start in H.nodes:
        extend([start])
    return sorted(found)


def qubit_load(instances) -> Counter:
    """How many instances each qubit belongs to, in any role."""
    return Counter(q for inst in instances for q in inst.qubits)


def min_circuits(instances) -> int:
    """Lower bound on the circuits: a qubit cannot be in two instances at once."""
    load = qubit_load(instances)
    return max(load.values()) if load else 0


def ancilla_coverage(instances) -> Counter:
    """How many times each qubit is benchmarked as an ancilla."""
    return Counter(a for inst in instances for a in inst.ancillas)


def select_chains(G, n_data=2, allowed_qubits=None, ancillas=None, cost=None, buffer=False,
                  restarts=None, seed=0, verbose=False):
    """Chains covering every ancilla-capable qubit, chosen so the packing is tightest.

    ``n_data = 2``: one triplet per candidate ancilla (``unique_ancilla_triples``).  A
    degree-3 ancilla keeps the neighbour pair that levels the qubit load best, ties broken by
    ``cost(Chain)`` (e.g. the calibration error budget).

    ``n_data > 2``: greedy cover - repeatedly grow a path through the least-connected
    uncovered ancilla so that it covers the most still-uncovered ancillas at the lowest
    load.  Ancillas are then measured at least once (not always exactly once), and a
    candidate that no chain of this length can reach is reported and skipped.

    ``restarts`` randomised orderings are tried; the one packing into fewest circuits wins.
    """
    H = region(G, allowed_qubits)
    cands = ancilla_candidates(H, ancillas=ancillas)
    cost = cost or (lambda chain: 0.0)
    rng = random.Random(seed)

    def score(instances):
        return (len(pack(instances, G, buffer=buffer)), min_circuits(instances), len(instances))

    if n_data == 2:
        options = {a: list(itertools.combinations(sorted(H[a]), 2)) for a in cands}
        costs = {(a, p): cost(Chain((p[0], a, p[1]))) for a in cands for p in options[a]}

        def build(order):
            load, chosen = Counter(), []
            for a in order:
                load[a] += 1
                p = min(options[a], key=lambda p: (load[p[0]] + load[p[1]],
                                                   max(load[p[0]], load[p[1]]),
                                                   costs[(a, p)], p))
                load[p[0]] += 1
                load[p[1]] += 1
                chosen.append(Chain((p[0], a, p[1])))
            return sorted(chosen)

        first = sorted(cands, key=lambda a: (len(options[a]), a))
        orders = [first] + [rng.sample(cands, len(cands)) for _ in range(64 if restarts is None else restarts)]
        unreachable = []
    else:
        orders = [None] * (8 if restarts is None else max(1, restarts))

        def build(_):
            return _cover(H, n_data, cands, cost, rng)

    best, best_score, unreachable = None, None, []
    for order in orders:
        result = build(order)
        chains, missed = (result, []) if n_data == 2 else result
        s = score(chains)
        if best_score is None or s < best_score:
            best, best_score, unreachable = chains, s, missed
    if verbose:
        cover = ancilla_coverage(best)
        print(f"{len(best)} chains of {n_data} data qubits | {len(cover)}/{len(cands)} candidate "
              f"ancillas covered (max {max(cover.values(), default=0)}x) | qubit load "
              f"{dict(sorted(Counter(qubit_load(best).values()).items()))} | "
              f"{best_score[0]} circuits (lower bound {best_score[1]})")
        if unreachable:
            print(f"  no chain of this length reaches ancillas {unreachable}")
    return best


def _cover(H, n_data, cands, cost, rng, attempts=24):
    length = 2 * n_data - 1
    load, uncovered, chains, unreachable = Counter(), set(cands), [], []
    while uncovered:
        anchor = min(uncovered, key=lambda a: (H.degree(a), rng.random()))
        best = None
        for _ in range(attempts):
            left = rng.randrange(1, length - 1, 2)      # the anchor sits at an odd index
            path = _grow(H, anchor, left, length - 1 - left, load, uncovered, rng)
            if path is None:
                continue
            chain = Chain(path)
            s = (-len(set(chain.ancillas) & uncovered), sum(load[q] for q in chain.qubits),
                 cost(chain))
            if best is None or s < best[0]:
                best = (s, chain)
        if best is None:
            uncovered.discard(anchor)
            unreachable.append(anchor)
            continue
        chain = best[1]
        chains.append(chain)
        load.update(chain.qubits)
        uncovered -= set(chain.ancillas)
    return sorted(set(chains)), sorted(unreachable)


def _grow(H, anchor, left, right, load, uncovered, rng, budget=5000, role_period=2):
    """Self-avoiding path with ``left`` steps before and ``right`` after the anchor.

    Depth-first with backtracking (a pure greedy walk dead-ends at chip edges), trying first
    the neighbours that land an uncovered qubit in a slot that matters, then the least
    loaded.  ``role_period=2`` cares only about the ancilla slots (every second position, as
    a chain needs); ``role_period=1`` cares about every position, which is what an
    ancilla-free line wants.
    """
    steps = [0] * left + [1] * right
    path, used, calls = [anchor], {anchor}, [0]

    def extend(i):
        calls[0] += 1
        if i == len(steps):
            return True
        if calls[0] > budget:
            return False
        side = steps[i]
        end = path[0] if side == 0 else path[-1]
        distance = i + 1 if side == 0 else i - left + 1
        wanted = distance % role_period == 0                # an ancilla slot, or any slot
        options = sorted((v for v in H[end] if v not in used),
                         key=lambda v: (-(wanted and v in uncovered), load[v], rng.random()))
        for v in options:
            used.add(v)
            path.insert(0, v) if side == 0 else path.append(v)
            if extend(i + 1):
                return True
            used.discard(v)
            path.pop(0) if side == 0 else path.pop()
        return False

    return path if extend(0) else None


def select_direct_chains(G, n_qubits=3, allowed_qubits=None, buffer=False, restarts=None,
                         seed=0, attempts=24, verbose=False):
    """Ancilla-free chains (``Direct``) covering the region - the direct reference.

    Every qubit of the region ends up in at least one of them, so the reference covers the
    same hardware the measurement-based chains do.  To compare a *specific* chain against its
    own qubits, use ``Direct.from_chain(chain)`` instead: same path, every qubit carrying data.
    """
    H = region(G, allowed_qubits)
    rng = random.Random(seed)
    best, best_score = None, None
    for _ in range(4 if restarts is None else max(1, restarts)):
        load, uncovered, chains = Counter(), set(H.nodes), []
        while uncovered:
            anchor = min(uncovered, key=lambda q: (H.degree(q), rng.random()))
            pick = None
            for _ in range(attempts):
                left = rng.randrange(0, n_qubits)
                path = _grow(H, anchor, left, n_qubits - 1 - left, load, uncovered, rng,
                             role_period=1)
                if path is None:
                    continue
                chain = Direct(path)
                score = (-len(set(chain.qubits) & uncovered), sum(load[q] for q in chain.qubits))
                if pick is None or score < pick[0]:
                    pick = (score, chain)
            if pick is None:                      # no chain of this length reaches the qubit
                uncovered.discard(anchor)
                continue
            chains.append(pick[1])
            load.update(pick[1].qubits)
            uncovered -= set(pick[1].qubits)
        chains = sorted(set(chains))
        score = (len(pack(chains, G, buffer=buffer)), len(chains))
        if best_score is None or score < best_score:
            best, best_score = chains, score
    if verbose:
        covered = {q for chain in best for q in chain.qubits}
        print(f"{len(best)} direct chains of {n_qubits} qubits | "
              f"{len(covered)}/{H.number_of_nodes()} qubits covered | {best_score[0]} circuits")
    return best


# --------------------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------------------
PACKING_STRATEGIES = ("saturation_largest_first", "largest_first", "smallest_last",
                      "independent_set")


def pack(instances, G=None, buffer=False, max_per_batch=None, strategies=None):
    """Partition instances into the fewest circuits of simultaneously-runnable instances.

    Two instances share a circuit only if they are qubit-disjoint; ``buffer=True`` also
    forbids them from being *coupled* (a one-qubit moat against measurement cross-talk).
    Every colouring heuristic is tried; fewest circuits wins, then the most even split -
    evenness is part of the experiment, because how many instances share a circuit sets how
    long their data qubits idle during feed-forward.  ``max_per_batch`` splits each colour
    class into equal chunks.
    """
    instances = list(instances)
    if not instances:
        return []
    if buffer and G is None:
        raise ValueError("buffer=True needs the coupling graph")
    conflict = nx.Graph()
    conflict.add_nodes_from(range(len(instances)))
    qsets = [set(inst.qubits) for inst in instances]
    for i, j in itertools.combinations(range(len(instances)), 2):
        clash = bool(qsets[i] & qsets[j])
        if not clash and buffer:
            clash = any(G.has_edge(u, v) for u in qsets[i] for v in qsets[j])
        if clash:
            conflict.add_edge(i, j)

    best = None
    for strategy in strategies or PACKING_STRATEGIES:
        try:
            colouring = nx.coloring.greedy_color(conflict, strategy=strategy)
        except Exception:
            continue
        groups = {}
        for i, colour in colouring.items():
            groups.setdefault(colour, []).append(instances[i])
        groups = [sorted(g) for _, g in sorted(groups.items())]
        if max_per_batch:
            chunked = []
            for g in groups:
                n_chunks = -(-len(g) // max_per_batch)
                size, extra = divmod(len(g), n_chunks)
                start = 0
                for k in range(n_chunks):
                    stop = start + size + (1 if k < extra else 0)
                    chunked.append(g[start:stop])
                    start = stop
            groups = chunked
        s = (len(groups), max(map(len, groups)) - min(map(len, groups)))
        if best is None or s < best[0]:
            best = (s, groups)
    return best[1]


def validate_batch(batch, G) -> list[str]:
    """Generic rules for one circuit: every coupler exists, instances are qubit-disjoint."""
    problems, owner = [], {}
    for inst in batch:
        for u, v in inst.couplers:
            if not G.has_edge(u, v):
                problems.append(f"{inst}: qubits {u} and {v} are not coupled")
        for q in inst.qubits:
            if q in owner:
                problems.append(f"qubit {q} is used by both {owner[q]} and {inst}")
            owner[q] = inst
    return problems
