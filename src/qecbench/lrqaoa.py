"""Linear-ramp QAOA on diagonal Ising Hamiltonians: schedule, energies and exact references.

Conventions (identical to those of the earlier campaigns, so their files compare directly):

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
    return float(bitstring_energies([bitstring], hamiltonian)[0])


def bitstring_energies(bitstrings, hamiltonian, chunk: int = 2048) -> np.ndarray:
    """``E`` of each bitstring (character ``i`` is qubit ``i``, spin ``1 - 2x``), vectorised.

    Terms of equal support size are evaluated together as the parity of a gathered bit array,
    so a sample costs a few array operations rather than a Python loop over its terms.
    """
    bitstrings = list(bitstrings)
    out = np.zeros(len(bitstrings))
    if not bitstrings:
        return out
    by_size = {}
    for term, c in hamiltonian.items():
        by_size.setdefault(len(term), ([], []))
        by_size[len(term)][0].append(list(term))
        by_size[len(term)][1].append(float(c))
    width = len(bitstrings[0])
    for start in range(0, len(bitstrings), chunk):
        block = bitstrings[start:start + chunk]
        bits = (np.frombuffer("".join(block).encode(), dtype=np.uint8).reshape(len(block), width) - 48)
        e = np.zeros(len(block))
        for size, (terms, weights) in by_size.items():
            if size == 0:
                e += sum(weights)
                continue
            parity = bits[:, np.array(terms)].sum(axis=2) & 1          # (bitstrings, terms)
            e += (1.0 - 2.0 * parity) @ np.array(weights)
        out[start:start + len(block)] = e
    return out


def energies(hamiltonian, n: int) -> np.ndarray:
    """Energy of every basis state; index ``k`` is bitstring ``format(k, f'0{n}b')``."""
    if n > 26:
        raise ValueError(f"{n} data qubits is too many for a dense energy table")
    index = np.arange(2 ** n, dtype=np.uint32 if n <= 32 else np.uint64)
    out = np.zeros(2 ** n)
    for term, c in hamiltonian.items():
        mask = 0
        for q in term:
            mask |= 1 << (n - 1 - q)                       # character q is the (n-1-q)-th bit
        parity = np.bitwise_count(index & np.uint32(mask)) & 1
        out += c * (1.0 - 2.0 * parity)
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
STORE_REFERENCE_QUBITS = 20      # exact references above this take seconds to minutes and are stored


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


def diagonal_statevector_energy(hamiltonian, n: int, depth: int, delta: float = 0.5) -> float:
    """Exact ``<H>`` by evolving the state directly: the cost layer is one diagonal phase.

    ``exp(-i gamma H)`` multiplies amplitude ``k`` by ``exp(-i gamma E_k)`` from the energy table,
    and the mixer is ``RX(-2 beta)`` on each qubit, applied in place on paired halves. Far
    faster than a gate-by-gate simulation for many-body terms (a weight-6 check is 11 gates),
    at the memory of one statevector plus the energy table (~0.8 GB at 25 qubits).
    """
    probabilities, table = _evolve(hamiltonian, n, depth, delta)
    return float(probabilities @ table)


def _evolve(hamiltonian, n, depth, delta):
    """Basis-state probabilities of the noiseless LR-QAOA and the energy table they refer to."""
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"{n} data qubits is past the {MAX_STATEVECTOR_QUBITS}-qubit exact limit")
    gammas, betas = angles(depth, delta)
    table = energies(hamiltonian, n)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * gamma * table)
        _mixer(psi, n, beta)
    return psi.real ** 2 + psi.imag ** 2, table


def ideal_energy_distribution(hamiltonian, n: int, depth: int, delta: float = 0.5):
    """Exact distribution of the energy of one noiseless LR-QAOA shot: ``(levels, probabilities)``.

    Everything a finite-shot question needs (how the mean of ``S`` shots spreads, how often it clears
    the random-guessing threshold) follows from it. It has one entry per distinct energy, so it is
    small even when the statevector is not, and it is **kept in the reference store** (``references``)
    next to ``<H>`` whatever the size: the statevector is evolved once and every later call reads the
    file. Past ``MAX_STATEVECTOR_QUBITS`` only a stored distribution can be returned (e.g. the histogram
    of a large noiseless simulation, stored by ``scripts/import_legacy_codes.py --references``);
    otherwise this raises.
    """
    from . import references

    stored = references.lookup(hamiltonian, n, depth, delta)
    if stored is not None and "distribution" in stored:
        levels, probabilities = map(np.array, zip(*stored["distribution"]))
        return levels, probabilities
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"no noiseless energy distribution for {n} data qubits at p = {depth}: the exact one "
                         f"stops at {MAX_STATEVECTOR_QUBITS} qubits and none is stored")
    probabilities, table = _evolve(hamiltonian, n, depth, delta)
    levels, inverse = np.unique(np.round(table, 9), return_inverse=True)
    probabilities = np.bincount(inverse, weights=probabilities, minlength=len(levels))
    references.store(hamiltonian, n, depth, delta, float(probabilities @ levels), "statevector",
                     distribution=[[float(e), float(q)] for e, q in zip(levels, probabilities)])
    return levels, probabilities


def _mixer(psi, n, beta, width=8):
    """``RX(-2 beta)`` on every qubit, as dense ``2^w x 2^w`` blocks applied by BLAS."""
    c, s = np.cos(beta), 1j * np.sin(beta)
    one = np.array([[c, s], [s, c]])
    q = 0
    while q < n:
        w = min(width, n - q)
        block = one
        for _ in range(w - 1):
            block = np.kron(block, one)
        left, right = 2 ** q, 2 ** (n - q - w)
        view = psi.reshape(left, 2 ** w, right)
        if left == 1:
            psi[:] = (block @ view[0]).reshape(-1)
        else:
            moved = np.ascontiguousarray(view.transpose(1, 0, 2)).reshape(2 ** w, -1)
            psi[:] = (block @ moved).reshape(2 ** w, left, right).transpose(1, 0, 2).reshape(-1)
        q += w


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
    * Otherwise, up to ``MAX_STATEVECTOR_QUBITS``, it is exact: direct NumPy evolution of the
      statevector (``diagonal_statevector_energy``, about 10x faster than applying the circuit gate by
      gate). Above ``STORE_REFERENCE_QUBITS`` that takes seconds to minutes, so the result is stored
      (``references``).
    * Above that there is no exact reference. A stored one (``references``) is used if present;
      otherwise the result is NaN with a warning. ``method="mps"`` runs Aer's
      **matrix-product-state** simulator explicitly, never building the 2**n amplitudes, which is
      exact only while the bond dimension keeps up with the entanglement - fine for chains and
      shallow circuits, not something to start silently on a 2D code at depth 10.

    ``method`` forces ``"free_fermions"``, ``"statevector"`` or ``"mps"``.
    """
    if method is None:
        chain = open_chain_couplings(hamiltonian, n)
        if chain is not None:
            method = "free_fermions"
        elif n <= MAX_STATEVECTOR_QUBITS:
            method = "statevector"
        else:
            from . import references

            stored = references.lookup(hamiltonian, n, depth, delta)
            if stored is not None:
                return float(stored["energy"])
            warnings.warn(
                f"no noiseless reference for {n} data qubits at p = {depth}: an exact statevector "
                f"stops at {MAX_STATEVECTOR_QUBITS} qubits and none is stored. Compute an estimate "
                f"explicitly (ideal_energy(..., method='mps', max_bond_dimension=...)) and keep it "
                f"with references.store(); until then r_ideal and r_ovl are NaN", stacklevel=2)
            return float("nan")
    if method == "free_fermions":
        couplings = open_chain_couplings(hamiltonian, n)
        if couplings is None:
            raise ValueError("free fermions need an open ZZ chain on qubits 0..n-1")
        return chain_energy_free_fermions(couplings, depth, delta)
    if method == "statevector":
        if n <= STORE_REFERENCE_QUBITS:
            return diagonal_statevector_energy(hamiltonian, n, depth, delta)
        from . import references

        stored = references.lookup(hamiltonian, n, depth, delta)
        if stored is not None and stored["method"] == "statevector":
            return float(stored["energy"])
        energy = diagonal_statevector_energy(hamiltonian, n, depth, delta)
        references.store(hamiltonian, n, depth, delta, energy, "statevector")
        return energy

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
    replaces the seeded 2000-sample bootstrap of the earlier analysis, whose
    mean for a triplet fluctuated around the exact 1/2 (0.5056 in the legacy files).
    """
    ham = primitive.hamiltonian
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    mean_e = sum(c for t, c in ham.items() if len(t) == 0)
    var_e = sum(c ** 2 for t, c in ham.items() if len(t) > 0)
    scale = abs(e_opt - e_max)
    return (mean_e - e_max) / (e_opt - e_max), np.sqrt(var_e / shots) / scale
