import json
from pathlib import Path

import numpy as np
import pytest

from qecbench import codes
from qecbench.backends import SimBackend
from qecbench.layout import surface_code_placements

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")
from qecbench import memory as mem  # noqa: E402


@pytest.mark.parametrize("d", [3, 5])
def test_code_typing_schedule_and_distance(d):
    code = codes.surface_code(d)
    assert mem.verify_code(code) == ((d * d - 1) // 2, (d * d - 1) // 2)
    assert all(mem.code_distance(code, R, basis) == d for R in (1, 2) for basis in "ZX")
    for check, slots in zip(code.checks, mem.cx_schedule(code)):
        assert sorted(q for q in slots if q is not None) == sorted(check)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_detectors_rebuilt_from_raw_bits_match_stim(basis):
    code = codes.surface_code(3)
    circuit = mem.stim_circuit(code, 3, basis, p_2q=0.02)
    raw = circuit.compile_sampler(seed=1).sample(400)
    dets, obs = circuit.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    raw = raw.astype(np.uint8)
    m = code.n_checks
    ours, logical = mem.detection_events(raw[:, :3 * m].reshape(-1, 3, m), raw[:, 3 * m:], code, basis)
    assert np.array_equal(ours, dets) and np.array_equal(logical, obs[:, 0])


def test_shot_keys_round_trip():
    syndromes, data = [[1, 0, 0, 1], [0, 1, 1, 0]], [1, 0, 1]
    key = mem.shot_key(syndromes, data)
    assert key == "1001 0110 101"
    s, dat = mem.shots_from_counts({key: 3}, rounds=2, n_checks=4, n_data=3)
    assert s.shape == (3, 2, 4) and np.array_equal(s[0], syndromes) and np.array_equal(dat[2], data)


def test_packed_shots_round_trip():
    rng = np.random.default_rng(0)
    syndromes, data = rng.integers(0, 2, (37, 3, 8), dtype=np.uint8), rng.integers(0, 2, (37, 9), dtype=np.uint8)
    packed = mem.pack_shots(syndromes, data)
    assert packed["shots"] == 37 and packed["bits_per_shot"] == 33
    s, d = mem.unpack_shots(packed, 3, 8, 9)
    assert np.array_equal(s, syndromes) and np.array_equal(d, data)


def test_error_per_round_recovers_the_model():
    rounds = np.array([1, 2, 3, 5, 7, 10])
    rates = (1 - 0.97 * (1 - 2 * 0.01) ** rounds) / 2
    eps, _, amplitude = mem.error_per_round(rounds, rates)
    assert eps == pytest.approx(0.01, rel=1e-3) and amplitude == pytest.approx(0.97, rel=1e-3)


def test_local_memory_loop(tmp_path):
    backend = SimBackend(topology="grid", qubits=121, seed=1)
    patch = surface_code_placements(backend.coupling_graph(), codes.surface_code(3))[0]
    plan = mem.plan(backend, [patch], rounds=[1, 3], bases=["Z", "X"], shots=500)
    assert len(plan["circuits"]) == 4 and plan["estimate"] is None
    with pytest.raises(ValueError, match="physical mcm placement"):
        mem.plan(backend, [type(patch)(patch.code)], rounds=[1])
    for noise, check in ((None, lambda r: r == 0), (mem.uniform_noise_model(1e-3, 1e-2, 1e-2), lambda r: 0 < r < 0.2)):
        manifest = mem.submit(plan, backend, manifest_dir=tmp_path / "m", noise_model=noise, seed=2)
        saved = mem.harvest(manifest, backend, data_dir=tmp_path / "r")
        (loaded_patch, by_basis), = mem.load_results(tmp_path / "r", backend.name, files={saved.split("/")[-1]}).items()
        assert loaded_patch == patch and set(by_basis) == {"Z", "X"}
        assert all(check(s["rate"]) and s["shots"] == 500 for by_R in by_basis.values() for s in by_R.values())


def test_plot_surface_code_draws_every_check():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Polygon, Wedge

    from qecbench.plotting import plot_surface_code

    code = codes.surface_code(5)
    ax, handles = plot_surface_code(code, bits="1" + "0" * 24)
    assert sum(isinstance(p, Polygon) for p in ax.patches) == 16 and sum(isinstance(p, Wedge) for p in ax.patches) == 8
    assert [h.get_label() for h in handles] == ["X check", "Z check", "logical Z", "logical X"]
    white = [p for p in ax.patches if p.get_facecolor()[:3] == (1.0, 1.0, 1.0)]
    assert len(white) == sum(1 for c in code.checks if 0 in c)        # the checks on data qubit 0 fire


def test_plot_patch_on_chip_marks_every_qubit_of_the_patch():
    import matplotlib
    matplotlib.use("Agg")

    from qecbench.layout import square_lattice_coordinates
    from qecbench.plotting import plot_patch_on_chip

    backend = SimBackend(topology="grid", qubits=121, seed=1)
    G = backend.coupling_graph()
    patch = surface_code_placements(G, codes.surface_code(3))[5]
    ax = plot_patch_on_chip(G, square_lattice_coordinates(G), patch, title="p")
    assert sorted(int(t.get_text()) for t in ax.texts) == sorted(patch.qubits)
    assert ax.get_title() == "p"


def test_loading_reads_the_harvested_rate_and_re_decodes_only_when_asked(tmp_path):
    backend = SimBackend(topology="grid", qubits=121, seed=1)
    patch = surface_code_placements(backend.coupling_graph(), codes.surface_code(3))[0]
    plan = mem.plan(backend, [patch], rounds=[1, 2], bases=["Z"], shots=200)
    manifest = mem.submit(plan, backend, manifest_dir=tmp_path / "m",
                          noise_model=mem.uniform_noise_model(1e-3, 1e-2, 1e-2), seed=3)
    saved = Path(mem.harvest(manifest, backend, data_dir=tmp_path / "r"))
    stored = {r["parameters"]["rounds"]: r["benchmark"]["logical_error_rate"]
              for r in json.loads(saved.read_text())["results"]}
    files = {saved.name}
    (by_basis,) = mem.load_results(tmp_path / "r", backend.name, files=files).values()
    assert {R: s["rate"] for R, s in by_basis["Z"].items()} == stored        # the harvested numbers, untouched
    (again,) = mem.load_results(tmp_path / "r", backend.name, files=files, decode_again=True).values()
    assert {R: s["rate"] for R, s in again["Z"].items()} == stored           # decoding again reproduces them
    (other,) = mem.load_results(tmp_path / "r", backend.name, files=files, p_2q_model=0.3).values()
    assert set(other["Z"]) == set(stored)                                    # other weights: decoded, may differ
