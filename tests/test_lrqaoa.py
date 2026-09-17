import numpy as np
import pytest

from qecbench import Chain
from qecbench.lrqaoa import (MAX_STATEVECTOR_QUBITS, angles, energies, ideal_circuit, ideal_energy,
                             ideal_probabilities, ideal_r, random_baseline)


def dense_reference(hamiltonian, n, depth, delta=0.5):
    """Independent statevector evolution, built with explicit Kronecker products.

    Qubit 0 is the most significant bit, i.e. the first character of the bitstring, matching
    ``energies``.  Used to pin the ordering of the Qiskit-based implementation.
    """
    gammas, betas = angles(depth, delta)
    diag = energies(hamiltonian, n)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    X = np.array([[0, 1], [1, 0]])
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * diag) * psi
        rx = np.cos(beta) * np.eye(2) + 1j * np.sin(beta) * X          # RX(-2 beta)
        for q in range(n):
            op = np.array([[1.0]])
            for k in range(n):
                op = np.kron(op, rx if k == q else np.eye(2))
            psi = op @ psi
    return np.abs(psi) ** 2


def notebook_noiseless_r(depth, delta=0.5):
    """Verbatim from benchmarking_ibm_auto.ipynb (section 11): exact two-spin LR-QAOA."""
    g = [k * delta / depth for k in range(1, depth + 1)]
    b = [(depth - k + 1) * delta / depth for k in range(1, depth + 1)]
    zz = np.array([1., -1., -1., 1.])
    psi = np.ones(4, dtype=complex) / 2
    for k in range(depth):
        psi = np.exp(-1j * g[k] * zz) * psi
        c, s = np.cos(b[k]), 1j * np.sin(b[k])
        psi = np.kron([[c, s], [s, c]], [[c, s], [s, c]]) @ psi
    return (1 - np.real(np.vdot(psi, zz * psi))) / 2


@pytest.mark.parametrize("depth", [1, 3, 6, 9, 12])
def test_triplet_ideal_matches_notebook(depth):
    assert ideal_r(Chain((0, 1, 2)), depth) == pytest.approx(notebook_noiseless_r(depth), abs=1e-12)


@pytest.mark.parametrize("hamiltonian, n", [
    ({(0, 1): 1.0, (1, 2): 1.0}, 3),                       # chain: symmetric under reversal
    ({(0, 1): 1.0, (1, 2): 2.5, (0,): -0.75}, 3),           # asymmetric: catches a bit-order slip
    ({(0, 1, 2): 1.0, (2, 3): 0.5}, 4),                     # a 3-body term, as a code stabiliser has
])
@pytest.mark.parametrize("depth", [1, 5])
def test_qiskit_probabilities_match_dense_reference(hamiltonian, n, depth):
    probs = ideal_probabilities(hamiltonian, n, depth)
    assert probs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(probs, dense_reference(hamiltonian, n, depth), atol=1e-12)


def test_ideal_probabilities_stays_on_plain_gates():
    """Guards the fast path: an evolution operator built by exponentiating a sparse matrix
    is ~2000x slower at 14 qubits and emits scipy warnings, so a warning here is a
    regression."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probs = ideal_probabilities(Chain(range(19)).hamiltonian, 10, 6)
    assert probs.sum() == pytest.approx(1.0)


def test_ideal_circuit_is_gadget_free_and_uses_zz_gates():
    qc = ideal_circuit({(0, 1): 1.0, (1, 2): 1.0}, 3, 2)
    assert qc.num_qubits == 3 and qc.num_clbits == 0      # data qubits only, no ancillas
    ops = qc.count_ops()
    assert not ops.get("measure")                          # a pure state, nothing measured
    assert ops["rzz"] == 4 and ops["rx"] == 6 and ops["h"] == 3      # 2 layers x 2 terms
    assert not {"cx", "rz"} & set(ops)                     # two-body terms need neither


def test_ideal_circuit_writes_many_body_terms_as_a_cx_ladder():
    ops = ideal_circuit({(0, 1, 2): 1.0}, 3, 1).count_ops()
    assert ops["cx"] == 4 and ops["rz"] == 1 and "rzz" not in ops


@pytest.mark.parametrize("n, depth", [(2, 4), (8, 3)])
def test_mps_reproduces_the_exact_energy(n, depth):
    """The MPS path must agree with the statevector where both can run."""
    hamiltonian = Chain(range(2 * n - 1)).hamiltonian
    exact = ideal_energy(hamiltonian, n, depth, method="statevector")
    with pytest.warns(UserWarning, match="matrix-product-state"):
        mps = ideal_energy(hamiltonian, n, depth, method="mps")
    assert mps == pytest.approx(exact, abs=1e-9)


def test_chains_use_exact_free_fermions_at_any_length():
    """An open ZZ chain is Gaussian under Jordan-Wigner: exact against the statevector where both
    run, and instant (with no MPS warning) far beyond it."""
    import warnings

    for n, depth, ham in [(2, 3, None), (7, 9, None), (5, 4, {(0, 1): 1.0, (1, 2): -0.7, (2, 3): 2.0, (3, 4): 0.3})]:
        ham = ham or Chain(range(2 * n - 1)).hamiltonian
        assert ideal_energy(ham, n, depth, method="free_fermions") == pytest.approx(
            ideal_energy(ham, n, depth, method="statevector"), abs=1e-10)
    n = 120
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ideal_r(Chain(range(2 * n - 1)), 50)
    assert 0.9 < r < 1.0


def test_non_chain_past_statevector_switches_to_mps_and_says_so():
    n = MAX_STATEVECTOR_QUBITS + 1
    ring = {**{(i, i + 1): 1.0 for i in range(n - 1)}, (0, n - 1): 1.0}      # closed: not a free-fermion chain
    with pytest.warns(UserWarning, match=f"{n} data qubits"):
        e = ideal_energy(ring, n, 1)
    assert -n < e < 0
    chain = Chain(range(2 * n - 1))
    # the distribution itself is 2**n numbers, so it is refused rather than attempted
    with pytest.raises(ValueError, match=f"2\\*\\*{n}"):
        ideal_probabilities(chain.hamiltonian, n, 2)


def test_angles_ramp():
    g, b = angles(4, 0.5)
    assert g == [0.125, 0.25, 0.375, 0.5] and b == [0.5, 0.375, 0.25, 0.125]


@pytest.mark.parametrize("n", [2, 3, 5])
def test_chain_energies_and_baseline(n):
    chain = Chain(range(2 * n - 1))
    e = energies(chain.hamiltonian, n)
    assert e.min() == chain.optimal_energy() == -(n - 1)
    assert e.max() == chain.max_energy() == n - 1
    mean, std = random_baseline(chain, shots=500)
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(np.sqrt((n - 1) / 500) / (2 * (n - 1)))


def test_chain_canonical_orientation_and_roles():
    c = Chain((7, 3, 1, 2, 5))
    assert c.qubits == (5, 2, 1, 3, 7)
    assert c.data_qubits == (5, 1, 7) and c.ancillas == (2, 3)
    assert c.hamiltonian == {(0, 1): 1.0, (1, 2): 1.0}
    assert Chain.from_roles([5, 1, 7], [2, 3]) == c
