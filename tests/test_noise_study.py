import numpy as np
import pytest

from qecbench import Chain
from qecbench.lrqaoa import energies, ideal_probabilities
from qecbench.noise_study import (Decay, accumulated_error, collapse, device_overlaps, device_run, fit_kappa,
                                  fit_kappa0, fit_lambda_eff, ideal_reference, lambda_from_cx, load_study,
                                  overlap_model, run_study, save_study, simulate_decay)


def density_matrix_reference(n, depth, lam, delta=0.5, per="edge"):
    """Exact density-matrix simulation on Aer.

    ``per="edge"``: each edge is one ``RZZ`` followed by ``depolarizing_error(lam, 2)``, the model
    of ``noise_study``. ``per="cx"``: each edge is ``CX RZ CX`` with the channel after every CX, the
    model of the earlier study.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    gammas = [(k + 1) * delta / depth for k in range(depth)]
    betas = [(depth - k) * delta / depth for k in range(depth)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n - 1):
            if per == "edge":
                qc.rzz(2 * gamma, i, i + 1)
            else:
                qc.cx(i, i + 1)
                qc.rz(2 * gamma, i + 1)
                qc.cx(i, i + 1)
        qc.rx(-2 * beta, range(n))
    qc.save_expectation_value(SparsePauliOp.from_sparse_list(
        [("ZZ", [i, i + 1], 1.0) for i in range(n - 1)], num_qubits=n), range(n))
    qc.save_probabilities(range(n))
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["rzz" if per == "edge" else "cx"])
    data = AerSimulator(method="density_matrix", noise_model=noise).run(qc, shots=1).result().data(0)
    alt = int(("01" * n)[:n], 2)
    probs = data["probabilities"]
    return ((n - 1) - data["expectation_value"]) / (2 * (n - 1)), probs[alt] + probs[(2 ** n - 1) ^ alt]


def absolute(decay, lam, observable):
    value = decay.overlap(lam, observable)[0][0]
    ideal, rand = getattr(decay, f"{observable}_ideal"), getattr(decay, f"{observable}_rand")
    return rand + value * (ideal - rand)


@pytest.mark.parametrize("n, depth", [(3, 2), (4, 1)])
def test_enumerated_decay_is_the_density_matrix(n, depth):
    """Every error configuration enumerated: the Pauli frame and the binomial sum over k
    together reproduce the density matrix to rounding."""
    decay = simulate_decay(n, depth, exhaustive_limit=10 ** 6)
    assert all(decay.exact) and max(decay.k) == decay.slots == depth * (n - 1)
    for lam in (0.003, 0.05, 0.3, 1.0):
        r, prob = density_matrix_reference(n, depth, lam)
        assert absolute(decay, lam, "r") == pytest.approx(r, abs=1e-12)
        assert absolute(decay, lam, "prob") == pytest.approx(prob, abs=1e-12)


def test_cnot_level_noise_is_a_per_edge_lambda():
    """A channel after each of an edge's two CNOTs is the per-edge model at lambda_from_cx."""
    decay = simulate_decay(3, 2, exhaustive_limit=10 ** 6)
    for lam_cx in (0.01, 0.2):
        r, prob = density_matrix_reference(3, 2, lam_cx, per="cx")
        assert absolute(decay, lambda_from_cx(lam_cx), "r") == pytest.approx(r, abs=1e-12)
        assert absolute(decay, lambda_from_cx(lam_cx), "prob") == pytest.approx(prob, abs=1e-12)


def test_sampled_decay_agrees_within_its_error_bars():
    decay = simulate_decay(5, 3, trajectories=512, seed=1)
    assert not all(decay.exact)
    for lam in (0.01, 0.05, 0.15):
        r, prob = density_matrix_reference(5, 3, lam)
        for observable, exact in (("r", r), ("prob", prob)):
            value, err, tail = decay.overlap(lam, observable)
            ideal, rand = getattr(decay, f"{observable}_ideal"), getattr(decay, f"{observable}_rand")
            target = (exact - rand) / (ideal - rand)
            assert abs(value[0] - target) < 4 * err[0] + tail[0] + 1e-6


@pytest.mark.parametrize("n, depth", [(6, 3), (9, 5)])
def test_noiseless_reference_matches_lrqaoa(n, depth):
    chain = Chain(range(2 * n - 1))
    probs = ideal_probabilities(chain.hamiltonian, n, depth)
    r = ((n - 1) - probs @ energies(chain.hamiltonian, n)) / (2 * (n - 1))
    alt = int(("01" * n)[:n], 2)
    assert ideal_reference(n, depth) == pytest.approx((r, probs[alt] + probs[(2 ** n - 1) ^ alt]))


def test_sampling_stops_once_the_signal_is_gone():
    decay = simulate_decay(6, 4, trajectories=64, seed=0)
    assert max(decay.k) < decay.slots                      # did not grind through all 20 slots
    s, e = decay.signal("r")
    assert s[0] == 1 and abs(s[-1]) < 2 * e[-1]
    assert np.all(np.diff(s[: len(s) // 2]) < 0)           # the early decay is monotone


def test_fits_recover_known_parameters():
    rng = np.random.default_rng(0)
    eps = np.logspace(-2, 1.5, 60)
    fit = fit_kappa(eps, overlap_model(eps, 0.36) + rng.normal(0, 1e-3, eps.size))
    assert fit["kappa"] == pytest.approx(0.36, rel=1e-2) and fit["r2"] > 0.999

    nqs = np.arange(5, 16)
    kappa0, stderr, rss = fit_kappa0(nqs, 3.5 / nqs)
    assert kappa0 == pytest.approx(3.5) and stderr == pytest.approx(0, abs=1e-12)

    depths = np.array([10, 15, 20, 30, 40, 50])
    assert accumulated_error(9, 10, 1e-2) == pytest.approx(0.9)          # N_edges p lambda
    ovl = overlap_model(accumulated_error(9, depths, 1.1e-2), 3.5 / 10)
    assert fit_lambda_eff(depths, ovl, 9, 3.5 / 10)["lambda_eff"] == pytest.approx(1.1e-2, rel=1e-6)


def test_study_cache_roundtrip(tmp_path):
    cache = tmp_path / "study.json"
    first = run_study([4], [1, 2], trajectories=32, cache=cache, verbose=False)
    again = load_study(cache)
    assert set(again) == {(4, 1), (4, 2)} and isinstance(again[(4, 2)], Decay)
    assert again[(4, 2)].r == first[(4, 2)].r
    reused = run_study([4], [1, 2], trajectories=32, cache=cache, verbose=False)
    assert reused[(4, 2)].seconds == first[(4, 2)].seconds             # read, not re-simulated
    points = collapse(reused, 4, [1e-3, 1e-2], "prob")
    assert points.shape == (5, 4)
    save_study(cache, reused)


def test_device_overlaps_normalise_against_the_exact_references():
    n = 6
    run = {p: (0.5 + 0.8 * (ideal_reference(n, p)[0] - 0.5), 0.1) for p in (2, 4)}
    depths, ovl = device_overlaps(run, n, observable="r")
    assert list(depths) == [2, 4] and ovl == pytest.approx([0.8, 0.8])


def test_fig5_device_runs_and_campaign_fits():
    """Fig. 5b reads converted runs, and Fig. 5c-5d reproduce the published fits x (3.51 / 3.69)."""
    import json
    from pathlib import Path

    from qecbench.analysis import load_results

    root = Path(__file__).resolve().parent.parent
    run = device_run(root / "data" / "results", "ibm_boston", "mcm", ["20260506_1537"], 10, [10, 20, 50])
    assert sorted(run) == [10, 20, 50] and all(0.5 < r < 1 and 0 <= prob <= 1 for r, prob in run.values())
    assert device_run(root / "data" / "results", "Helios-1E", "direct", ["20260507_0900"], 10, [3])[3][0] > 0.8

    kappa0 = json.loads((root / "data" / "noise_study" / "kappa_fits.json").read_text())["kappa_0"]["value"]

    def lam(backend, stamps, window):
        res = load_results(root / "data" / "results", backend, kind="mcm",
                           files={f"{s}_{backend}_chain_mcm.json" for s in stamps})
        out = {}
        for chain, by in res.items():
            n, ps = chain.n_data, np.array([p for p in window if p in by], dtype=float)
            ovl = np.array([by[int(p)]["r_ovl"] for p in ps])
            if (ovl > 0).sum() >= 2:                    # as the notebook: points at the random floor are dropped
                out[n] = fit_lambda_eff(ps[ovl > 0], ovl[ovl > 0], n - 1, kappa0 / n)["lambda_eff"]
        return out

    boston = lam("ibm_boston", ["20260827_1035"], [3, 5, 10])
    assert boston[10] == pytest.approx(0.95 * 2.330e-02, rel=0.02)       # published Fig. 5c MCM value
    phoenix = np.mean(list(lam("ibm_phoenix", ["20260904_1435"], [5, 10, 15, 20]).values()))
    assert phoenix == pytest.approx(0.95 * 8.140e-02, rel=0.02)           # published Fig. 5d mean
