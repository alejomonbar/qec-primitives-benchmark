"""Linear-ramp QAOA on diagonal Ising Hamiltonians: schedule, energies and exact references.

Conventions (identical to ``utils.py`` / ``benchmarking_ibm*.ipynb`` in the MCM repo):

* bitstring character ``i`` is data qubit ``i``; spin ``z_i = 1 - 2 x_i``;
* ``H = sum_t c_t prod_{i in t} z_i``;
* one layer is ``exp(-i gamma_k H)`` followed by ``RX(-2 beta_k)`` on every data qubit;
* ``r = (E - E_max) / (E_opt - E_max)``: 0 at the worst energy, 1 at the ground state.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np


def angles(depth: int, delta: float = 0.5):
    """Linear ramp: ``gamma_k = k delta / p``, ``beta_k = (p - k + 1) delta / p``."""
    gammas = [k * delta / depth for k in range(1, depth + 1)]
    betas = [(depth - k + 1) * delta / depth for k in range(1, depth + 1)]
    return gammas, betas


def energy(bitstring: str, hamiltonian) -> float:
    spins = [1 - 2 * int(b) for b in bitstring]
    return float(sum(c * np.prod([spins[i] for i in term]) for term, c in hamiltonian.items()))


def energies(hamiltonian, n: int) -> np.ndarray:
    """Energy of every basis state; index ``k`` is bitstring ``format(k, f'0{n}b')``."""
    if n > 26:
        raise ValueError(f"{n} data qubits is too many for a dense energy table")
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    spins = 1 - 2 * bits
    out = np.zeros(2 ** n)
    for term, c in hamiltonian.items():
        out += c * np.prod(spins[:, list(term)], axis=1)
    return out


def ideal_circuit(hamiltonian, n: int, depth: int, delta: float = 0.5):
    """The LR-QAOA circuit itself, on ``n`` data qubits and without the ancilla gadget.

    Each term ``c * Z_i Z_j ...`` of one layer is written directly, exactly, with the gate
    that performs it - no Trotter step is involved, because the terms of a diagonal Ising
    Hamiltonian all commute:

    * two data qubits -> ``RZZ(2 c gamma)``, since ``RZZ(t) = exp(-i t Z Z / 2)``;
    * one  data qubit -> ``RZ(2 c gamma)``;
    * more (a code stabiliser) -> CX ladder onto the last qubit, which then holds the parity
      of the term, one ``RZ(2 c gamma)``, and the ladder undone.

    Qubit indices are used as given, so nothing here depends on Qiskit's bit ordering.  This
    is the noiseless reference the device is compared against, and it deliberately shares
    nothing with the gadget circuits of ``circuits.py``.
    """
    from qiskit import QuantumCircuit

    gammas, betas = angles(depth, delta)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for term, coeff in hamiltonian.items():
            angle = 2 * coeff * gamma
            if len(term) == 2:
                qc.rzz(angle, term[0], term[1])
            elif len(term) == 1:
                qc.rz(angle, term[0])
            elif len(term) > 2:
                *rest, last = term
                for q in rest:
                    qc.cx(q, last)
                qc.rz(angle, last)
                for q in reversed(rest):
                    qc.cx(q, last)
        qc.rx(-2 * beta, range(n))
    return qc


MAX_STATEVECTOR_QUBITS = 25   # 2^25 amplitudes is 0.5 GB; beyond this the dense path is hopeless


def ideal_probabilities(hamiltonian, n: int, depth: int, delta: float = 0.5) -> np.ndarray:
    """Noiseless output distribution of the depth-``p`` LR-QAOA, from a Qiskit statevector.

    Index ``k`` is the bitstring ``format(k, f'0{n}b')`` in this package's convention -
    character ``i`` is data qubit ``i`` - which is the reverse of Qiskit's little-endian
    ordering, hence the reversed ``qargs``.

    Only defined up to ``MAX_STATEVECTOR_QUBITS``: the answer itself has ``2**n`` entries, so a
    wider chain cannot be returned in this form however it is computed.  ``ideal_energy`` is
    the way to get the reference for those, since it needs only ``<H>``.
    """
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(
            f"{n} data qubits: the distribution alone is 2**{n} numbers. Use ideal_energy() or "
            f"ideal_r(), which get the noiseless reference from a matrix-product-state "
            f"simulation without ever building it.")
    from qiskit.quantum_info import Statevector

    state = Statevector(ideal_circuit(hamiltonian, n, depth, delta))
    return np.asarray(state.probabilities(range(n - 1, -1, -1)))


def open_chain_couplings(hamiltonian, n: int):
    """Coupling of each bond ``(i, i+1)`` if ``hamiltonian`` is an open ZZ chain on ``0..n-1``, else None."""
    if len(hamiltonian) != n - 1 or n < 2:
        return None
    couplings = []
    for i in range(n - 1):
        c = hamiltonian.get((i, i + 1), hamiltonian.get((i + 1, i)))
        if c is None:
            return None
        couplings.append(float(c))
    return couplings


def chain_energy_free_fermions(couplings, depth: int, delta: float = 0.5) -> float:
    """Exact ``<H>`` of LR-QAOA on an open chain ``H = sum_i J_i Z_i Z_{i+1}``, at any length and depth.

    With a Hadamard on every qubit the circuit becomes ``|0...0>``, ``exp(-i gamma J_i X_i X_{i+1})``
    and ``exp(i beta Z_i)``, and ``<Z_i Z_{i+1}>`` becomes ``<X_i X_{i+1}>``. Under Jordan-Wigner,
    ``c_{2j} = Z..Z X_j`` and ``c_{2j+1} = Z..Z Y_j``, so ``Z_j = -i c_{2j} c_{2j+1}`` and
    ``X_j X_{j+1} = -i c_{2j+1} c_{2j+2}``: every gate is ``exp(theta c_a c_b)``, a free-fermion
    rotation, which moves the Majorana operators by ``c_a -> cos 2theta c_a + sin 2theta c_b``,
    ``c_b -> cos 2theta c_b - sin 2theta c_a``. The state stays Gaussian, so the ``2n x 2n``
    covariance matrix ``M_kl = <i c_k c_l>`` carries everything. Cost ``O(p n^3)``, no
    approximation.
    """
    n = len(couplings) + 1
    gammas, betas = angles(depth, delta)
    heis = np.eye(2 * n)                                  # c_k -> sum_l heis[k, l] c_l

    def rotate(a, b, theta):
        # U_1^+ ... U_t^+ c U_t ... U_1: the newest gate acts on c first, so its rotation
        # multiplies the accumulated map from the left
        c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
        row_a, row_b = heis[a].copy(), heis[b].copy()
        heis[a], heis[b] = c2 * row_a + s2 * row_b, c2 * row_b - s2 * row_a

    for gamma, beta in zip(gammas, betas):
        for j, J in enumerate(couplings):                 # exp(-i gamma J X_j X_{j+1}) = exp(-gamma J c c)
            rotate(2 * j + 1, 2 * j + 2, -gamma * J)
        for j in range(n):                                # exp(i beta Z_j) = exp(beta c c)
            rotate(2 * j, 2 * j + 1, beta)
    m0 = np.zeros((2 * n, 2 * n))
    for j in range(n):                                    # |0>: <Z_j> = 1, so <i c_2j c_2j+1> = -1
        m0[2 * j, 2 * j + 1], m0[2 * j + 1, 2 * j] = -1.0, 1.0
    m = heis @ m0 @ heis.T
    return float(sum(J * -m[2 * j + 1, 2 * j + 2] for j, J in enumerate(couplings)))


def ideal_energy(hamiltonian, n: int, depth: int, delta: float = 0.5, method: str | None = None,
                 max_bond_dimension: int | None = None) -> float:
    """``<H>`` of the noiseless LR-QAOA - the only thing the reference ``r`` needs.

    * An open ZZ chain (any couplings) is solved exactly as free fermions at any length and depth
      (``chain_energy_free_fermions``), in milliseconds.
    * Otherwise, up to ``MAX_STATEVECTOR_QUBITS``, the exact distribution is contracted with the
      energy table.
    * Above that, the circuit runs on Aer's **matrix-product-state** simulator and ``<H>`` is read
      off directly, never building the 2**n amplitudes. A warning says so, because MPS is only
      exact while the bond dimension keeps up with the entanglement, and a deep enough ramp will
      eventually outgrow it.

    ``method`` forces ``"free_fermions"``, ``"statevector"`` or ``"mps"``.
    """
    if method is None:
        chain = open_chain_couplings(hamiltonian, n)
        method = ("free_fermions" if chain is not None else
                  "statevector" if n <= MAX_STATEVECTOR_QUBITS else "mps")
    if method == "free_fermions":
        couplings = open_chain_couplings(hamiltonian, n)
        if couplings is None:
            raise ValueError("free fermions need an open ZZ chain on qubits 0..n-1")
        return chain_energy_free_fermions(couplings, depth, delta)
    if method == "statevector":
        return float(ideal_probabilities(hamiltonian, n, depth, delta) @ energies(hamiltonian, n))

    warnings.warn(
        f"noiseless reference for {n} data qubits: using a matrix-product-state simulation"
        + ("" if n <= MAX_STATEVECTOR_QUBITS else
           f", since an exact statevector stops at {MAX_STATEVECTOR_QUBITS} qubits")
        + ". MPS is exact only while the bond dimension keeps up with the entanglement, so treat"
          " deep, wide chains with care"
        + (f"; bond dimension capped at {max_bond_dimension}" if max_bond_dimension else ""),
        stacklevel=2)
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator

    qc = ideal_circuit(hamiltonian, n, depth, delta)
    observable = SparsePauliOp.from_sparse_list(
        [("Z" * len(term), list(term), coeff) for term, coeff in hamiltonian.items()],
        num_qubits=n)
    qc.save_expectation_value(observable, range(n))
    options = {"method": "matrix_product_state"}
    if max_bond_dimension:
        options["matrix_product_state_max_bond_dimension"] = max_bond_dimension
    result = AerSimulator(**options).run(qc, shots=1).result()
    return float(result.data(0)["expectation_value"])


@lru_cache(maxsize=512)
def _ideal_r_cached(terms, n, depth, delta, e_opt, e_max):
    ham = dict(terms)
    e = ideal_energy(ham, n, depth, delta)          # statevector, or MPS past 25 data qubits
    return (e - e_max) / (e_opt - e_max)


def ideal_r(primitive, depth: int, delta: float = 0.5) -> float:
    """Approximation ratio the primitive would reach on a noiseless device."""
    terms = tuple(sorted(primitive.hamiltonian.items()))
    return _ideal_r_cached(terms, primitive.n_data, depth, delta,
                           primitive.optimal_energy(), primitive.max_energy())


def random_baseline(primitive, shots: int):
    """Mean and shot-noise std of ``r`` for uniformly random bitstrings - exact.

    Distinct ``Z`` products are orthogonal under the uniform distribution, so
    ``E[H] = 0`` and ``Var[H] = sum c_t^2``; ``shots`` samples average that down.  This
    replaces the seeded 2000-sample bootstrap of ``utils.statistical_analysis``, whose
    mean for a triplet fluctuated around the exact 1/2 (0.5056 in the legacy files).
    """
    ham = primitive.hamiltonian
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    mean_e = sum(c for t, c in ham.items() if len(t) == 0)
    var_e = sum(c ** 2 for t, c in ham.items() if len(t) > 0)
    scale = abs(e_opt - e_max)
    return (mean_e - e_max) / (e_opt - e_max), np.sqrt(var_e / shots) / scale
