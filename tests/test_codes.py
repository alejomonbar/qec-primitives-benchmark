import json

import networkx as nx
import numpy as np
import pytest

from qecbench import CodePatch, codes, from_dict, references
from qecbench.analysis import load_results
from qecbench.backends import QuantinuumBackend, code_instances
from qecbench.backends.aer import AerBackend
from qecbench.circuits import compact_index
from qecbench.code_programs import cx_rounds, guppy_source, qiskit_program
from qecbench.experiment import build_plan, harvest, submit
from qecbench.layout import spread_selection, square_lattice_coordinates, surface_code_placements
from qecbench.lrqaoa import energies


@pytest.fixture(autouse=True)
def private_reference_store(tmp_path, monkeypatch):
    monkeypatch.setenv("QECBENCH_REFERENCES", str(tmp_path / "refs.json"))


def square_grid(rows=12, cols=10):
    G = nx.Graph()
    G.add_edges_from([(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)])
    G.add_edges_from([(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)])
    return G


# -- structures ------------------------------------------------------------------------------
@pytest.mark.parametrize("d, checks, weights", [(3, 8, {2: 4, 4: 4}), (5, 24, {2: 8, 4: 16}),
                                                (9, 80, {2: 16, 4: 64})])
def test_surface_code_structure(d, checks, weights):
    code = codes.surface_code(d)
    assert code.n_data == d * d and code.n_checks == checks and code.check_weights == weights
    assert code.optimal_energy() == -checks and code.max_energy() == checks


def test_counts_follow_the_checks():
    sc9 = codes.surface_code(9)
    assert sc9.counts("mcm") == {"two_qubit_gates": 288, "mid_circuit_measurements": 80}
    assert sc9.counts("direct", depth=2) == {"two_qubit_gates": 2 * 416, "mid_circuit_measurements": 0}
    cc = codes.color_code(7)
    assert cc.n_data == 37 and cc.counts("mcm")["mid_circuit_measurements"] == 18


@pytest.mark.parametrize("name, n_data", [("BB18", 18), ("BB24", 24), ("GB16", 16)])
def test_qldpc_checks_are_independent_and_merged(name, n_data):
    code = codes.bivariate_bicycle(name)
    assert code.n_data == n_data and len(set(code.checks)) == code.n_checks
    assert all(w in (1.0, 2.0) for w in code.weights)
    assert code.optimal_energy() == -sum(code.weights)


def test_from_checks_merges_duplicates_and_round_trips(tmp_path):
    code = codes.from_checks("toy", 5, [(0, 1), (1, 2, 3, 4), (0, 1)], note="hand made")
    assert code.hamiltonian == {(0, 1): 2.0, (1, 2, 3, 4): 1.0} and code.info == {"note": "hand made"}
    assert codes.load(code.save(tmp_path / "toy.json")) == code
    with pytest.raises(ValueError, match="invalid check"):
        codes.from_checks("bad", 3, [(0, 3)])
    assert codes.build("qldpc", "BB18") == codes.bivariate_bicycle("BB18")


def test_ground_state_energy_is_exact():
    code = codes.color_code(3)
    E = energies(code.hamiltonian, code.n_data)
    assert code.optimal_energy() == E.min() and code.max_energy() == E.max()
    frustrated = codes.from_checks("triangle", 3, [(0, 1), (1, 2), (0, 2)])      # no all-odd assignment
    assert frustrated.optimal_energy() == energies(frustrated.hamiltonian, 3).min() == -1


def test_load_reads_a_hamiltonian_in_every_form(tmp_path):
    ref = codes.color_code(3)
    txt = tmp_path / "color_code_d3_hamiltonian.txt"
    txt.write_text("#n_qubits:7\n" + "\n".join(str(list(c)) for c in ref.checks) + "\n[0, 1] 0.5\n")
    from_txt = codes.load(txt)
    assert from_txt.name == "color_code_d3" and from_txt.n_data == 7
    assert from_txt.hamiltonian == {**ref.hamiltonian, (0, 1): 0.5}
    legacy = tmp_path / "20260316_101939_Helios-1E_cc_normal_nq7_depth15.json"
    legacy.write_text(json.dumps({"hamiltonian": {"num_nodes": 7, "hamiltonian_couplings": [list(c) for c in ref.checks],
                                                  "hamiltonian_weights": [1] * ref.n_checks}}))
    assert codes.load(legacy, name="cc3", family="color_code").checks == ref.checks
    assert codes.load(ref.save(tmp_path / "cc.json")) == ref
    ham = codes.from_hamiltonian("h", {(0, 2): 1.0, (1, 2, 3): -0.5})
    assert ham.n_data == 4 and ham.optimal_energy() == -1.5

# -- schedules -------------------------------------------------------------------------------
@pytest.mark.parametrize("code, batches", [(codes.surface_code(5), 4), (codes.color_code(5), 3),
                                           (codes.bivariate_bicycle("BB18"), 7)])
def test_schedule_batches_are_disjoint_and_cover_every_check(code, batches):
    sched = codes.schedule(code)
    assert sched.n_batches == batches
    assert sorted(c for b in sched.batches for c in b) == list(range(code.n_checks))
    for batch in sched.batches:
        support = [q for c in batch for q in code.checks[c]]
        assert len(support) == len(set(support))
    idle = sum(code.n_data - sum(len(code.checks[c]) for c in b) for b in sched.batches)
    assert sched.idle_qubit_steps() == idle and sched.peak_qubits("direct") == code.n_data


def test_max_parallel_cuts_batches():
    code = codes.surface_code(5)
    sched = codes.schedule(code, max_parallel=3)
    assert sched.max_batch == 3 and sched.peak_qubits("mcm") == 28
    assert sorted(c for b in sched.batches for c in b) == list(range(code.n_checks))


def test_helios_schedule_fits_the_free_qubits():
    cc11 = codes.color_code(11)
    assert cc11.n_data == 91
    sched = QuantinuumBackend("Helios-1").code_schedule(cc11)
    assert sched.max_batch <= 7 and sched.peak_qubits("mcm") <= 98
    capped = QuantinuumBackend("Helios-1", max_parallel=4).code_schedule(codes.surface_code(5))
    assert capped.max_batch == 4 and QuantinuumBackend("Helios-1E").zones == 8


# -- instances and records --------------------------------------------------------------------
def test_code_patch_labels_and_round_trip():
    code = codes.surface_code(3)
    mcm, direct = code_instances(code)
    assert mcm.logical and mcm.qubits == tuple(range(17)) and direct.qubits == tuple(range(9))
    assert mcm.tag == "surface_d3_mcm" and mcm.optimal_energy() == -8
    assert [t.ancilla for t in mcm.terms] == list(range(9, 17)) and all(t.ancilla is None for t in direct.terms)
    assert from_dict(json.loads(json.dumps(mcm.to_dict()))) == mcm
    assert mcm != direct and len({mcm, direct, CodePatch(code)}) == 2
    with pytest.raises(ValueError, match="ancillas"):
        CodePatch(code, ancillas=range(3))


def test_helios_rejects_physical_code_patches():
    code = codes.surface_code(3)
    backend = QuantinuumBackend("Helios-1")
    physical = CodePatch(code, range(10, 19), range(30, 38))
    assert "logical labels" in backend.validate([physical], "mcm")[0]
    assert backend.validate([code_instances(code, ("mcm",))[0]], "mcm") == []


def test_reference_store_round_trip():
    H = codes.surface_code(3).hamiltonian
    assert references.lookup(H, 9, 3, 0.5) is None
    references.store(H, 9, 3, 0.5, -1.25, "test", note="x")
    entry = references.lookup(H, 9, 3, 0.5)
    assert entry["energy"] == -1.25 and entry["method"] == "test" and entry["note"] == "x"
    assert references.lookup(H, 9, 4, 0.5) is None
    assert references.key(dict(reversed(list(H.items()))), 9, 3, 0.5) == references.key(H, 9, 3, 0.5)


# -- programs --------------------------------------------------------------------------------
def test_cx_rounds_take_the_jth_qubit_of_every_check():
    assert cx_rounds([(0, 1, 2, 3), (4, 5)]) == [[(0, 0), (4, 1)], [(1, 0), (5, 1)], [(2, 0)], [(3, 0)]]


@pytest.mark.parametrize("kind", ["mcm", "direct"])
def test_qiskit_program_gate_counts(kind):
    code = codes.surface_code(3)
    sched = codes.schedule(code)
    qc = qiskit_program(code, sched, depth=2, kind=kind)
    ops = qc.count_ops()
    counts = code.counts(kind, depth=2)
    assert ops["cx"] == counts["two_qubit_gates"]
    assert qc.num_qubits == sched.peak_qubits(kind)
    if kind == "mcm":
        assert ops["measure"] == counts["mid_circuit_measurements"] + 9 and ops["reset"] == 2 * code.n_checks
    else:
        assert ops["measure"] == 9 and "reset" not in ops


def test_guppy_source_mirrors_the_schedule():
    code = codes.color_code(3)
    sched = codes.schedule(code)
    src = guppy_source(code, sched, depth=3, kind="mcm")
    assert src.count("@guppy") == sched.n_batches + 1
    assert src.count("    batch_0(qs, ") == 3
    assert src.count("measure_array(anc)") == sched.n_batches
    assert src.count("cx(") == code.counts("mcm")["two_qubit_gates"]      # written once per batch
    direct = guppy_source(code, sched, depth=3, kind="direct")
    assert "anc" not in direct and direct.count("cx(") == code.counts("direct")["two_qubit_gates"]


def test_guppy_code_programs_compile():
    pytest.importorskip("guppylang")
    code = codes.from_checks("toy", 5, [(0, 1), (1, 2, 3), (3, 4)], weights=[1, 0.5, 2])
    for kind in ("mcm", "direct"):
        program = QuantinuumBackend("Helios-1").build(code_instances(code, (kind,)), 2, 0.5, kind)
        assert program.definition.compile() is not None
        assert program.peak_qubits == (5 + 2 if kind == "mcm" else 5) and "toy-" + kind in program.name


def test_local_helios_code_loop_matches_the_exact_reference(tmp_path):
    code = codes.from_checks("toy", 5, [(0, 1), (1, 2, 3), (3, 4)], weights=[1, 0.5, 2])
    backend = QuantinuumBackend("Helios-1", local=True, seed=7)
    plan = build_plan(backend, code_instances(code), depths=[2], shots=4000)
    harvest(submit(plan, backend, manifest_dir=tmp_path / "m"), backend, data_dir=tmp_path / "r")
    for kind in ("mcm", "direct"):
        res = load_results(tmp_path / "r", backend.name, kind=kind)
        (patch, by_depth), = res.items()
        assert patch == CodePatch(code, kind=kind)
        s = by_depth[2]
        assert abs(s["r"] - s["r_ideal"]) < 5 * s["r_err"]


# -- square-lattice placements ------------------------------------------------------------------
def test_surface_code_placements_on_a_square_grid():
    G = square_grid()
    coords = square_lattice_coordinates(G)
    assert coords[0] == (0, 0) and coords[11] == (1, 1)
    assert square_lattice_coordinates(nx.cycle_graph(7)) is None
    d3 = surface_code_placements(G, codes.surface_code(3))
    d5 = surface_code_placements(G, codes.surface_code(5))
    assert len(d3) == 48 and len(d5) == 8
    for patch in d3 + d5:
        assert AerBackend(G).validate([patch], "mcm") == []
        assert len(set(patch.qubits)) == len(patch.qubits)
    with pytest.raises(ValueError, match="surface_code"):
        surface_code_placements(G, codes.color_code(3))


def test_placed_surface_code_circuit_uses_the_couplers():
    G = square_grid()
    patch = surface_code_placements(G, codes.surface_code(3))[0]
    qc = AerBackend(G).build([patch], 1, 0.5, "mcm")
    _, layout = compact_index([patch])                    # circuit index -> physical qubit
    two_qubit = [tuple(layout[qc.find_bit(q).index] for q in inst.qubits) for inst in qc.data if len(inst.qubits) == 2
                 and inst.operation.name not in ("measure", "barrier")]
    assert two_qubit and all(G.has_edge(*pair) for pair in two_qubit)
    assert np.isclose(sum(t.weight for t in patch.terms), 8)


def test_spread_selection_keeps_the_ends_of_the_ranking():
    items = [f"p{i}" for i in range(48)]
    scores = [(7 * i) % 48 for i in range(48)]
    chosen = spread_selection(items, scores, 12)
    picked = [score for _, score in chosen]
    assert picked == sorted(picked) and picked[0] == 0 and picked[-1] == 47 and len(chosen) <= 12
    assert len(spread_selection(items[:5], scores[:5], 8)) == 5


# -- legacy code runs ----------------------------------------------------------------------------
def legacy_code_file(code, backend="Helios-1", shots=None):
    samples = {"101000101": 30, "010111010": 15, "000000000": 5}
    return {"metadata": {"backend": backend, "task_id": "job-1", "timestamp": "2026-03-09T16:31:34"},
            "parameters": {"depth": 3, "delta": 0.5, "num_data_qubits": code.n_data, "data_qubits": list(range(code.n_data)),
                           "ancillary_qubits": [], **({"shots": shots} if shots else {})},
            "hamiltonian": {"num_nodes": code.n_data, "hamiltonian_couplings": [list(c) for c in reversed(code.checks)],
                            "hamiltonian_weights": [1] * code.n_checks},
            "samples": samples, "energy_analysis": {"approximation_ratio": 0.0}}


def test_convert_legacy_code_runs():
    from qecbench.analysis import convert_legacy_code, instance_from_record

    code = codes.surface_code(3)
    stamp, record = convert_legacy_code(legacy_code_file(code), "20260309_163134_Helios-1_sc_MCM_nq9_depth3.json")
    patch = instance_from_record(record)
    assert stamp == "20260309_163134" and patch == CodePatch(code, kind="mcm") and patch.code == code
    b = record["benchmark"]
    assert b["program"] == "qaoa_mcm_sc" and b["kind_source"] == "file name" and not record["metadata"]["simulated"]
    energy = (30 * -8 + 15 * energies(code.hamiltonian, 9)[int("010111010"[::-1], 2)] + 5 * 8) / 50
    assert np.isclose(b["r"], (energy - 8) / -16)

    _, direct = convert_legacy_code(legacy_code_file(code, "Helios-1"), "20260306_162159_Helios-1E_sc_nq9_depth15.json")
    assert direct["parameters"]["kind"] == "direct" and direct["benchmark"]["kind_source"].startswith("inferred")
    assert direct["metadata"]["backend"] == "Helios-1E" and direct["metadata"]["simulated"]
    assert "metadata says Helios-1" in direct["benchmark"]["backend_note"]

    bb = codes.bivariate_bicycle("BB18")
    legacy = legacy_code_file(bb)
    legacy["samples"] = {"0" * 18: 10}
    _, rec = convert_legacy_code(legacy, "20260702_0853_Helios-1_qldpc_BB18_normal_nq18_depth3_combined.json")
    assert instance_from_record(rec).code.name == "BB18" and rec["benchmark"]["r"] == 0.0

    assert convert_legacy_code(legacy_code_file(code), "20260306_152722_qasm_simulator_sc_nq9_depth5.json") is None
    wrong = legacy_code_file(code)
    wrong["hamiltonian"]["hamiltonian_couplings"][0] = [0, 1]
    assert convert_legacy_code(wrong, "20260309_163134_Helios-1_sc_MCM_nq9_depth3.json") is None


def test_figure6_runs_match_the_published_bars():
    from qecbench.analysis import load_results

    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "results"
    # the imported campaigns themselves: a later run of the same structure and depth must not stand in
    published = {("Helios-1", "surface_code", "mcm", "surface_d9", 5, "20260309_1631"): 0.48,
                 ("Helios-1E", "color_code", "direct", "color_d5", 20, None): 0.76,
                 ("Helios-1", "qldpc", "mcm", "BB48", 10, None): 0.5631}
    for (backend, family, kind, name, p, stamp), r in published.items():
        files = {f.name for f in (root / backend / family / kind).glob(f"{stamp or ''}*.json")}
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            res = load_results(root, backend, kind=kind, files=files)
        (by_depth,) = [v for k, v in res.items() if k.code.name == name]
        assert round(by_depth[p]["r"], 4) == r


# -- shot budgets ---------------------------------------------------------------------------------
def test_ideal_energy_distribution_is_exact():
    from qecbench.lrqaoa import ideal_energy, ideal_energy_distribution

    code = codes.color_code(3)
    levels, probabilities = ideal_energy_distribution(code.hamiltonian, code.n_data, 3)
    assert np.isclose(probabilities.sum(), 1) and len(levels) <= 2 * code.n_checks + 1
    assert np.isclose(levels @ probabilities, ideal_energy(code.hamiltonian, code.n_data, 3))


def test_separation_from_random():
    from qecbench.shots import ideal_r_distribution, sample_distribution, separation_probability, shots_to_separate

    patch = CodePatch(codes.surface_code(3))
    values, probabilities = ideal_r_distribution(patch, 3)
    grid = [5, 10, 20, 30]
    separated = separation_probability(patch, values, probabilities, grid, n_boot=20_000, seed=1)
    assert np.all(np.diff(separated) >= -0.01) and shots_to_separate(grid, separated) == 10     # Fig. 7a, d = 3
    counts = {"101000101": 60, "010111010": 30, "000000000": 10}
    r, q = sample_distribution(counts, patch)
    assert np.isclose(q @ r, (0.6 * -8 + 0.3 * energies(patch.hamiltonian, 9)[int("010111010", 2)] + 0.1 * 8 - 8) / -16)
    ovl, _ = sample_distribution(counts, patch, depth=3, quantity="r_ovl")
    assert np.allclose(ovl, (r - 0.5) / (ideal_r_distribution(patch, 3)[0] @ probabilities - 0.5))



def test_convert_legacy_ibm_code_run_onto_its_patch():
    from qecbench.analysis import convert_legacy_code, instance_from_record

    code = codes.surface_code(3)
    legacy = legacy_code_file(code, backend="ibm_phoenix")
    name = "20260909_1631_ibm_phoenix-scanbest_sc_MCM_nq9_depth3.json"
    assert convert_legacy_code(legacy, name) is None                          # no placement, no physical qubits
    (patch,) = surface_code_placements(square_grid(), code, anchors=[(7, 4)])
    stamp, rec = convert_legacy_code(legacy, name, placement=patch)
    assert rec["metadata"]["backend"] == "ibm_phoenix" and rec["benchmark"]["run_tag"] == "scanbest"
    assert instance_from_record(rec) == patch and not patch.logical and "backend_note" not in rec["benchmark"]
    assert rec["benchmark"]["program"] == "square-lattice MCM LR-QAOA (legacy builder)" and not rec["metadata"]["simulated"]


def test_shots_to_rank_and_white_noise():
    from qecbench.shots import (ideal_r_distribution, moments, random_r_distribution, shots_to_rank,
                                white_noise_distribution)

    assert shots_to_rank(0.6, 0.2, 0.7, 0.2, z=3) == pytest.approx(9 * 0.08 / 0.01)
    patch = CodePatch(codes.color_code(3), kind="direct")
    values, ideal = ideal_r_distribution(patch, 3)
    rand_values, rand = random_r_distribution(patch)
    assert np.allclose(values, rand_values) and np.isclose(moments(rand_values, rand)[0], 0.5)
    full, none = (moments(values, white_noise_distribution(ideal, rand, a))[0] for a in (1.0, 0.0))
    assert np.isclose(full, values @ ideal) and np.isclose(none, 0.5)
