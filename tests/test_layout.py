from collections import Counter

import networkx as nx
import pytest

from qecbench import Chain
from qecbench.backends.iqm import FF_GROUPS
from qecbench.layout import (ancilla_candidates, enumerate_chains, load_cached_graph, min_circuits,
                             pack, restrict_to_groups, select_chains, select_direct_chains,
                             validate_batch)


@pytest.fixture(scope="module")
def garnet():
    return load_cached_graph("iqm_garnet")


def test_triple_counts_match_legacy_notebook(garnet):
    # benchmarking_iqm_auto.ipynb: "54 geometric triples, 35 executable"
    assert len(enumerate_chains(garnet, 2)) == 54
    assert len(enumerate_chains(restrict_to_groups(garnet, FF_GROUPS["iqm_garnet"]), 2)) == 35


def test_unique_ancilla_triplets(garnet):
    triples = select_chains(garnet, 2, restarts=8)
    cands = ancilla_candidates(garnet)
    assert sorted(t.ancillas[0] for t in triples) == cands          # each exactly once
    batches = pack(triples, garnet)
    assert sum(map(len, batches)) == len(triples)
    assert len(batches) >= min_circuits(triples)
    for b in batches:
        assert validate_batch(b, garnet) == []


def test_heavy_hex_triplets_reach_load_bound():
    G = nx.convert_node_labels_to_integers(nx.hexagonal_lattice_graph(3, 3))
    triples = select_chains(G, 2, restarts=16)
    assert len(pack(triples, G)) == min_circuits(triples)


@pytest.mark.parametrize("n_data", [3, 4])
def test_chain_cover(garnet, n_data):
    chains = select_chains(garnet, n_data, restarts=2)
    assert all(c.n_data == n_data for c in chains)
    covered = Counter(a for c in chains for a in c.ancillas)
    assert set(covered) == set(ancilla_candidates(garnet))
    for c in chains:
        assert validate_batch([c], garnet) == []


@pytest.mark.parametrize("n_qubits", [2, 3, 5])
def test_direct_chain_cover(garnet, n_qubits):
    chains = select_direct_chains(garnet, n_qubits, restarts=2)
    assert all(c.kind == "direct" and c.ancillas == () and c.n_data == n_qubits for c in chains)
    assert {q for c in chains for q in c.qubits} == set(garnet.nodes)   # every qubit covered
    for c in chains:
        assert validate_batch([c], garnet) == []                        # every bond is a coupler


def test_buffer_packing_keeps_moat(garnet):
    triples = select_chains(garnet, 2, restarts=2)
    for b in pack(triples, garnet, buffer=True):
        qs = [set(t.qubits) for t in b]
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                assert not any(garnet.has_edge(u, v) for u in qs[i] for v in qs[j])


def test_validate_catches_overlap_and_missing_coupler(garnet):
    problems = validate_batch([Chain((1, 2, 5)), Chain((5, 6, 7)), Chain((1, 3, 8))], garnet)
    assert any("used by both" in p for p in problems)
    assert any("not coupled" in p for p in problems)
