import json
from pathlib import Path

import networkx as nx
import pytest

from qecbench import Chain, Direct
from qecbench.analysis import instance_from_record, load_results, parse_filename, records_in


def first_record(path):
    """The first result inside a run file (or a bare single-result file)."""
    return records_in(json.loads(Path(path).read_text()))[0]
from qecbench.backends import AerBackend, IQMBackend
from qecbench.experiment import build_plan, harvest, print_plan, submit
from qecbench.layout import restrict_to_groups, select_chains

LEGACY = Path("/Users/alejomonbar/Documents/GitHub/Benchmarking-Mid-circuit-measurement/Data")


def run_loop(backend, instances, tmp_path, depths=(2, 5)):
    plan = build_plan(backend, instances, depths=list(depths), shots=2000)
    print_plan(plan, backend)
    manifest = submit(plan, backend, manifest_dir=tmp_path / "m")
    saved = harvest(manifest, backend, data_dir=tmp_path / "r")
    # one run file per (structure, kind), not one per instance and depth
    assert len(saved) == len({(i.structure, i.kind) for i in instances})
    assert harvest(manifest, backend, data_dir=tmp_path / "r") == []      # idempotent
    stored = sum(len(json.loads(Path(f).read_text())["results"]) for f in saved)
    assert stored == len(set(instances)) * len(depths)
    return plan, manifest


def test_aer_chain_and_direct_loop(tmp_path):
    """Both kinds in one plan: they are packed apart but submitted in the same jobs."""
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4))
    backend = AerBackend(graph=G, seed=3)
    chains = select_chains(G, 3, restarts=2)
    direct = sorted({Direct.from_chain(c) for c in chains})
    assert all(d.n_data == c.n_data for c, d in zip(chains, direct))   # same logical circuit
    plan, manifest = run_loop(backend, chains + direct, tmp_path)
    assert set(plan["batches"]) == {"mcm", "direct"}
    assert {t["kind"] for t in plan["tasks"]} == {"mcm", "direct"}

    by_kind = {kind: load_results(tmp_path / "r", backend.name, kind=kind, manifest_path=manifest)
               for kind in ("mcm", "direct")}
    assert set(by_kind["mcm"]) == set(chains)
    assert set(by_kind["direct"]) == set(direct)

    # results are filed by <backend>/<structure>/<kind>, and nothing is left loose
    root = tmp_path / "r" / backend.name / "chain"
    assert {p.name for p in root.iterdir()} == {"mcm", "direct"}
    assert list((tmp_path / "r").glob("*.json")) == []
    assert (tmp_path / "m" / backend.name).is_dir()          # manifests too
    # the whole campaign is two files, each carrying its own run header
    files = sorted((tmp_path / "r").rglob("*.json"))
    assert len(files) == 2
    header = json.loads(files[0].read_text())["run"]
    assert header["backend"] == backend.name and header["shots"] == 2000
    assert header["manifest"] == Path(manifest).name and header["results"] > 1
    for res in by_kind.values():
        for by_depth in res.values():
            for s in by_depth.values():
                assert abs(s["r"] - s["r_ideal"]) < 6 * s["r_err"] + 1e-3


def test_iqm_local_triplet_loop(tmp_path):
    backend = IQMBackend("iqm_garnet", local=True, seed=5)
    # the "_sim" name must not break the device lookups (graph cache, groups, pricing)
    assert backend.name == "iqm_garnet_sim" and backend.device_name == "iqm_garnet"
    assert backend.feedforward_groups() and backend.simulated
    G = restrict_to_groups(backend.coupling_graph(), backend.feedforward_groups())
    triples = select_chains(G, 2, allowed_qubits=[8, 13, 14, 15, 18, 19], restarts=2)
    plan, manifest = run_loop(backend, triples + [Direct.from_chain(t) for t in triples], tmp_path)
    assert plan["estimate"]["unit"] == "USD"
    chain_rec, direct_rec = (first_record(sorted((tmp_path / "r").rglob(f"*_chain_{kind}.json"))[0])
                             for kind in ("mcm", "direct"))
    assert chain_rec["parameters"]["kind"] == "mcm"
    assert chain_rec["parameters"]["num_data_qubits"] == 2          # a triplet: d1, a, d2
    assert len(chain_rec["parameters"]["ancillary_qubits"]) == 1
    assert direct_rec["parameters"]["kind"] == "direct"
    assert direct_rec["parameters"]["num_data_qubits"] == 2         # same 2-spin problem, no ancilla
    assert direct_rec["parameters"]["ancillary_qubits"] == []
    assert "benchmark" in chain_rec and "benchmark" in direct_rec


def test_ibm_local_fake_backend_loop(tmp_path):
    fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from qecbench.backends import IBMBackend

    backend = IBMBackend(backend=fake_provider.FakeFez(), local=True, seed=7)
    assert backend.name == "fake_fez_sim" and backend.simulated       # never mistaken for hardware
    assert "NoiseModel.from_backend(fake_fez)" in backend.noise_description
    G = backend.coupling_graph()
    triples = select_chains(G, 2, allowed_qubits=range(10), restarts=2)
    chains = select_chains(G, 3, allowed_qubits=range(10), restarts=1)
    run_loop(backend, triples + [Direct.from_chain(t) for t in triples], tmp_path / "tri", depths=(2,))
    plan, manifest = run_loop(backend, chains, tmp_path / "chain", depths=(2,))
    assert plan["estimate"]["unit"] == "QPU s" and plan["estimate"]["total"] > 0
    res = load_results(tmp_path / "chain" / "r", backend.name, kind="mcm", n_data=3)
    rs = [s["r"] for by in res.values() for s in by.values()]
    assert len(set(rs)) > 1                                   # no shared random stream
    assert all(0.5 < r < 1.0 for r in rs)                     # noisy, but far from random

    rec = first_record(sorted((tmp_path / "chain" / "r").rglob("*.json"))[0])
    assert rec["metadata"]["simulated"] is True and rec["benchmark"]["simulated"] is True
    assert "fake_fez" in rec["metadata"]["noise_model"]
    assert rec["metadata"]["backend"] == "fake_fez_sim"


def test_noisy_simulator_runs_without_an_account(tmp_path):
    """The whole loop on a synthetic chip: no vendor SDK credentials involved."""
    from qecbench.backends import SimBackend

    backend = SimBackend(qubits=18, seed=2)
    G = backend.coupling_graph()
    assert G.number_of_nodes() == 18 and max(dict(G.degree).values()) <= 3   # heavy-hex-like
    assert backend.simulated and "depolarizing" in backend.noise_description

    chains = select_chains(G, 2, restarts=1)[:4]
    run_loop(backend, chains + sorted({Direct.from_chain(c) for c in chains}),
             tmp_path, depths=(2,))
    res = load_results(tmp_path / "r", backend.name, kind="mcm", n_data=2)
    assert res and all(0.5 < s["r"] <= 1.0 for by in res.values() for s in by.values())

    rec = first_record(sorted((tmp_path / "r").rglob("*.json"))[0])
    assert rec["metadata"]["simulated"] is True
    assert "depolarizing" in rec["metadata"]["noise_model"]

    noiseless = SimBackend(qubits=9, p_1q=0, p_2q=0, readout=0, seed=3)
    assert "0.00e+00 per 2q gate" in noiseless.noise_description
    with pytest.raises(TypeError, match="unknown noise parameter"):
        SimBackend(qubits=9, p_3q=0.1)


def test_iqm_rejects_mcm_chains():
    backend = IQMBackend("iqm_garnet", local=True)
    with pytest.raises(ValueError, match="one controller"):
        build_plan(backend, [Chain((1, 2, 5, 6, 7))], depths=[1], shots=10)


def test_filename_roundtrip():
    assert parse_filename("20260902_1629_ibm_phoenix_tri_0_10_11_mcm_nq2_depth12.json") == {
        "backend": "ibm_phoenix", "qubits": (0, 10, 11), "kind": "mcm", "n_data": 2, "depth": 12,
        "family": "chain", "legacy": True}
    assert parse_filename("20260916_101010_iqm_garnet_direct_4_5_6_direct_nq3_depth3.json") == {
        "backend": "iqm_garnet", "qubits": (4, 5, 6), "kind": "direct", "n_data": 3, "depth": 3,
        "family": "direct", "legacy": False}
    # kinds this package no longer builds still parse, so old campaigns keep loading
    assert parse_filename("20260911_101110_ibm_boston_tri_4_5_6_mcm_reset_nq2_depth3.json")["kind"] == "mcm_reset"


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy MCM repository not present")
def test_legacy_ibm_triplets_load():
    res = load_results(LEGACY, "ibm_phoenix", kind="mcm",
                       manifest_path=LEGACY / "manifests/20260911_102009_ibm_phoenix_mcm_triples.json")
    assert len(res) == 120
    s = next(iter(res.values()))
    assert set(s) == {3, 6, 9, 12} and s[3]["r_rand"] == pytest.approx(0.5)


def test_legacy_1d_chain_files_convert_to_this_convention(tmp_path):
    """A qubit-reuse MCM chain (no ancillas stored) becomes a Chain with logical ancilla labels; a
    `normal` file becomes the direct kind; derived numbers are recomputed from the samples."""
    from qecbench.analysis import convert_legacy_chain, run_filename, save_run
    from qecbench.lrqaoa import ideal_r

    samples = {"0101": 30, "1010": 10, "0011": 10}
    base = {"metadata": {"backend": "Helios-1", "task_id": "t1", "timestamp": "2026-09-15T10:35:57"},
            "parameters": {"depth": 3, "delta": 0.5, "num_data_qubits": 4, "data_qubits": [0, 1, 2, 3],
                           "ancillary_qubits": []},
            "hamiltonian": {"hamiltonian_couplings": [[0, 1], [1, 2], [2, 3]], "hamiltonian_weights": [1, 1, 1]},
            "samples": samples, "random_baseline": {"r_approximation_ratio": 0.4996}}
    stamp, mcm = convert_legacy_chain(base, "20260915_1035_Helios-1_1d_MCM_nq4_depth3.json")
    assert stamp == "20260915_1035" and mcm["parameters"]["kind"] == "mcm"
    assert mcm["parameters"]["ancillary_qubits"] == [4, 5, 6]
    assert mcm["benchmark"]["ancilla_labels"].startswith("logical")
    assert mcm["metadata"]["timestamp"] == "2026-09-15T10:35:57"          # original, not today's
    chain = instance_from_record(mcm)
    assert isinstance(chain, Chain) and chain.data_qubits == (0, 1, 2, 3)
    r = (40 * 1.0 + 10 * 1 / 3) / 50                                    # 0101/1010 optimal; 0011 cuts one bond of three
    assert mcm["benchmark"]["r"] == pytest.approx(r)
    assert mcm["benchmark"]["r_ideal"] == pytest.approx(ideal_r(chain, 3))
    assert mcm["benchmark"]["r_ovl"] == pytest.approx((r - 0.5) / (ideal_r(chain, 3) - 0.5))

    _, direct = convert_legacy_chain(base, "20260915_1035_Helios-1_1d_normal_nq4_depth3.json")
    assert direct["parameters"]["kind"] == "direct" and direct["parameters"]["ancillary_qubits"] == []
    assert convert_legacy_chain(base, "20260504_0852_ibm_boston_1d_MCM-DD_nq4_depth3.json") is None
    _, pinned = convert_legacy_chain(base, "20260827_1035_ibm_boston_1dpin_MCM_nq4_depth3.json")
    assert pinned["metadata"]["backend"] == "Helios-1" and pinned["benchmark"]["layout"] == "pinned"
    assert "energy_values" not in pinned["energy_analysis"]
    code = {**base, "hamiltonian": {"hamiltonian_couplings": [[0, 1, 2, 3]], "hamiltonian_weights": [1]}}
    assert convert_legacy_chain(code, "20260309_1703_Helios-1_sc_normal_nq4_depth3.json") is None

    path = save_run(tmp_path / "Helios-1" / "chain" / "mcm" / run_filename("Helios-1", "chain", "mcm", stamp), [mcm])
    loaded = load_results(tmp_path, "Helios-1", kind="mcm")
    assert set(loaded) == {chain} and loaded[chain][3]["r"] == pytest.approx(r) and path.endswith("_chain_mcm.json")


def test_imported_helios_chains_are_in_the_repository():
    root = Path(__file__).resolve().parent.parent / "data" / "results"
    res = load_results(root, "Helios-1", kind="mcm")
    assert sorted(c.n_data for c in res) == [20, 30, 40, 50]
    assert all(set(by) == {3, 5, 10} and by[3]["shots"] == 50 for by in res.values())


@pytest.mark.parametrize("device, stamps, n, median", [
    ("iqm_garnet", ["20260813_0729"], 35, 0.564),
    ("iqm_emerald", ["20260813_0722", "20260813_0723"], 104, 0.672),
    ("ibm_kingston", ["20260813_1637"], 148, 0.686),
    ("ibm_boston", ["20260814_0818"], 148, 0.778),
    ("ibm_phoenix", ["20260902_1629"], 120, 0.558),
])
def test_fig3c_triplet_data_matches_the_published_panel(device, stamps, n, median):
    import numpy as np

    root = Path(__file__).resolve().parent.parent / "data" / "results"
    res = load_results(root, device, kind="mcm", n_data=2,
                       files={f"{stamp}_{device}_chain_mcm.json" for stamp in stamps})
    r = [by[3]["r"] for by in res.values() if 3 in by]
    assert len(r) == n and round(float(np.median(r)), 3) == median
    assert all(set(by) == {3, 6, 9, 12} for by in res.values())       # every depth of the campaign came along


def test_fig3ef_best_triplets_and_helios_emulator_runs():
    root = Path(__file__).resolve().parent.parent / "data" / "results"
    expected = {("ibm_boston", "20260814_0818"): (136, 143, 142), ("ibm_kingston", "20260813_1637"): (147, 148, 149),
                ("iqm_garnet", "20260813_0729"): (15, 14, 18)}
    for (device, stamp), winner in expected.items():
        res = load_results(root, device, kind="mcm", n_data=2, files={f"{stamp}_{device}_chain_mcm.json"})
        assert max((by[3]["r_ovl"], c.qubits) for c, by in res.items())[1] == winner

    stamps = ["20260224_153017", "20260224_153157", "20260224_153300", "20260224_153359", "20260224_154357"]
    helios = load_results(root, "Helios-1E", kind="mcm", files={f"{s}_Helios-1E_chain_mcm.json" for s in stamps})
    (chain, by_depth), = helios.items()
    assert chain.qubits == (0, 2, 1) and sorted(by_depth) == [3, 5, 7, 10, 12]
    assert by_depth[3]["r"] == pytest.approx(0.89)
    record = first_record(root / "Helios-1E" / "chain" / "mcm" / f"{stamps[0]}_Helios-1E_chain_mcm.json")
    assert record["metadata"]["simulated"] is True and "emulator" in record["metadata"]["noise_model"]


def test_fig4_serialized_helios_runs_cost_more_per_edge():
    import json

    import numpy as np
    from qecbench.noise_study import fit_lambda_eff

    root = Path(__file__).resolve().parent.parent
    kappa0 = json.loads((root / "data" / "noise_study" / "kappa_fits.json").read_text())["kappa_0"]["value"]

    def lam(stamps):
        res = load_results(root / "data" / "results", "Helios-1", kind="mcm",
                           files={f"{s}_Helios-1_chain_mcm.json" for s in stamps})
        out = {}
        for chain, by in res.items():
            n, ps = chain.n_data, np.array([3.0, 5.0, 10.0])
            out[n] = fit_lambda_eff(ps, [by[int(p)]["r_ovl"] for p in ps], n - 1, kappa0 / n)["lambda_eff"]
        return out

    serial = lam(["20260309_0726", "20260525_0955", "20260526_0735"])
    august = lam(["20260825_1505", "20260825_1540", "20260825_1541", "20260825_1542", "20260825_1543", "20260825_1606"])
    parallel = lam(["20260909_1423", "20260909_1509", "20260909_1521"])
    assert set(serial) == set(august) == set(parallel) == {30, 40, 50}
    assert all(serial[n] > 4 * parallel[n] for n in serial)
    assert all(august[n] > 2.4 * parallel[n] for n in august)     # same chains, two weeks apart: the ordering
    assert serial[50] == pytest.approx(0.95 * 3.349e-02, rel=0.02)      # published value x (3.51 / 3.69)
