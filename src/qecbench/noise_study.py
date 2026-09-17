"""Depolarizing noise on LR-QAOA for a 1D chain: kappa(N_q), kappa_0 and lambda_eff.

Not part of the benchmark. It backs the assumption used to turn a measured overlap into an
effective error rate. Put a two-qubit depolarizing channel of strength ``lambda`` after the ZZ
interaction of every edge in every layer. Then the normalised overlaps of an ``N_q``-spin chain at
depth ``p`` fall on one curve::

    r_ovl, rho_ovl = 2 ** (-kappa * eps_acc),      eps_acc = N_edges p lambda

with ``N_edges = N_q - 1``. For the approximation ratio ``kappa_r = kappa_0 / N_q``. For the
probability of the optimum ``kappa_rho`` hardly depends on size. Fitting ``lambda`` to a device's
overlaps, with ``kappa`` held fixed, gives its ``lambda_eff``.

``lambda`` is an error **per edge interaction**, not per gate. How an edge is built is left to
the implementation: the direct circuit spends two CNOTs on it, the mid-circuit-measurement circuit
spends CZs to an ancilla, a measurement and feed-forward. ``lambda_eff`` therefore puts both on the
same scale, and their difference is what the extra resources of one cost relative to the other.
The earlier study put ``depolarizing_error(lambda_CX, 2)`` after each of the two CNOTs instead.
That is exactly this model with ``lambda = 1 - (1 - lambda_CX)**2`` (``lambda_from_cx``).

Why this is fast
----------------
A density-matrix simulation with thousands of shots for every (depth, lambda) is slow. Two exact
identities and a regrouping remove nearly all of that work.

1. **Errors only negate angles.** The channel leaves the pair alone with probability ``1 - q``,
   ``q = 15/16 lambda``, and otherwise applies one of the 15 non-identity Paulis uniformly. Pushed
   to the end of the circuit, a Pauli with an X on one qubit of a later ``RZZ`` flips that angle's
   sign, a Z under a later ``RX`` flips that one, and what arrives at the end only relabels the
   measured bits. Every noisy run is therefore a pure-state LR-QAOA with some angles negated: a
   statevector, never a density matrix.
2. **One simulation serves every lambda.** Given that exactly ``k`` of the ``p N_edges`` edge
   interactions failed, where and how they failed does not depend on ``lambda``. So::

       <O>(lambda) = sum_k Binomial(k; p N_edges, q(lambda)) m_k

   and the conditional means ``m_k`` are estimated once per ``(N_q, p)``.
3. (For the CNOT-level model.) A two-qubit depolarizing channel commutes with any unitary on the
   same pair, so ``CX . D . RZ . CX . D`` is ``RZZ`` followed by one channel, which is how the
   conversion above is exact.

``m_k`` is exact (every configuration enumerated) while that is cheaper than sampling, and a
Monte-Carlo mean over sampled configurations otherwise. Its standard error is carried through to
every overlap. As ``k`` grows ``m_k`` falls to the random value, so sampling stops once the
remaining signal is below the noise. The probability left beyond the last ``k`` is reported as
``tail``, and no signal is assumed for it.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from math import comb
from pathlib import Path

import numpy as np

OBSERVABLES = ("r", "prob")
CHUNK = 5            # qubits per dense block of the mixer; 32x32 blocks suit BLAS best


def accumulated_error(edges: int, depth, lam):
    """``eps_acc = N_edges p lambda``: the error accumulated over every edge interaction.

    For a chain ``N_edges = N_q - 1``.
    """
    return edges * np.asarray(depth, dtype=float) * np.asarray(lam, dtype=float)


def overlap_model(eps, kappa):
    return 2.0 ** (-kappa * np.asarray(eps, dtype=float))


def slot_error_probability(lam):
    """Probability that an edge interaction suffers a non-identity Pauli."""
    return 15 / 16 * np.asarray(lam, dtype=float)


def lambda_from_cx(lam_cx):
    """Per-edge ``lambda`` equivalent to ``depolarizing_error(lam_cx, 2)`` after each of the two
    CNOTs that build the edge's ZZ rotation (exact, since the channel commutes with those gates)."""
    return 1 - (1 - np.asarray(lam_cx, dtype=float)) ** 2


# ======================================================================================
# Pure-state kernel
# ======================================================================================
class _Chain:
    """Lookup tables for an ``n``-spin chain ``H = sum_i Z_i Z_{i+1}``.

    Bit ``i`` of a basis index, counted from the most significant end, is qubit ``i``, as in
    ``lrqaoa.energies``.
    """

    def __init__(self, n: int, chunk: int = CHUNK):
        if n < 2:
            raise ValueError("a chain needs at least two spins")
        self.n, self.bonds = n, n - 1
        z = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
        self.zz = (z[:, :-1] * z[:, 1:]).T.astype(np.float64)             # (bonds, 2^n)
        alt = int(("01" * n)[:n], 2)
        self.optimal = np.array([alt, (2 ** n - 1) ^ alt])                 # the two Neel states
        self.weights = 1 << np.arange(n - 1, -1, -1)

        self.blocks = []                                                   # (start, width, bonds)
        for start in range(0, n, chunk):
            width = min(chunk, n - start)
            inner = list(range(start, start + width - 1))
            zl = 1 - 2 * ((np.arange(2 ** width)[:, None] >> np.arange(width - 1, -1, -1)) & 1)
            table = (zl[:, :-1] * zl[:, 1:]).T.astype(np.float64)          # (width-1, 2^width)
            self.blocks.append((start, width, inner, table))
        self.seams = [start - 1 for start, *_ in self.blocks[1:]]          # bonds across blocks

    def layer(self, psi, gamma, beta, bond_signs, mixer_signs):
        """``exp(-i gamma sum_b s_b ZZ_b)`` then ``prod_q RX(-2 beta s_q)``, for a batch of states."""
        m, n = psi.shape[0], self.n
        same = np.exp(-1j * gamma * bond_signs[:, self.seams])             # (m, seams)
        for j, b in enumerate(self.seams):
            f = np.empty((m, 2, 2), dtype=complex)
            f[:, 0, 0] = f[:, 1, 1] = same[:, j]
            f[:, 0, 1] = f[:, 1, 0] = np.conj(same[:, j])
            psi.reshape(m, 2 ** b, 2, 2, -1)[...] *= f[:, None, :, :, None]

        c, s = np.cos(beta), np.sin(beta)
        for start, width, inner, table in self.blocks:
            block = np.ones((m, 1, 1), dtype=complex)
            for q in range(start, start + width):
                g = np.empty((m, 2, 2), dtype=complex)
                g[:, 0, 0] = g[:, 1, 1] = c
                g[:, 0, 1] = g[:, 1, 0] = 1j * s * mixer_signs[:, q]
                d = block.shape[1] * 2
                block = (block[:, :, None, :, None] * g[:, None, :, None, :]).reshape(m, d, d)
            if inner:
                block = block * np.exp(-1j * gamma * (bond_signs[:, inner] @ table))[:, None, :]
            left, right = 2 ** start, 2 ** (n - start - width)
            view = psi.reshape(m, left, 2 ** width, right)
            if left == 1:
                np.copyto(view, (block @ view.reshape(m, 2 ** width, right))[:, None])
            else:
                moved = np.ascontiguousarray(view.transpose(0, 2, 1, 3)).reshape(m, 2 ** width, -1)
                out = (block @ moved).reshape(m, 2 ** width, left, right)
                np.copyto(view, out.transpose(0, 2, 1, 3))
        return psi

    def measure(self, psi, flips):
        """``r`` and the probability of the optimum, per state, after the final bit flips."""
        probs = psi.real ** 2 + psi.imag ** 2
        zz = probs @ self.zz.T                                             # (m, bonds)
        sign = 1 - 2 * (flips[:, :-1] ^ flips[:, 1:])
        energy = (zz * sign).sum(axis=1)
        r = (self.bonds - energy) / (2 * self.bonds)
        shift = (flips * self.weights).sum(axis=1)
        prob = probs[np.arange(len(psi))[:, None], self.optimal[None, :] ^ shift[:, None]].sum(1)
        return r, prob


def _schedule(depth, delta):
    gammas = [(k + 1) * delta / depth for k in range(depth)]
    betas = [(depth - k) * delta / depth for k in range(depth)]
    return gammas, betas


def _frames(chain, depth, paulis):
    """Angle signs and final bit flips for each error configuration.

    ``paulis`` is ``(T, depth, bonds)``: 0 for no error, else a 4-bit code ``x0 z0 x1 z1`` (low bit
    first) of the Pauli on the edge's two qubits, applied after that edge's ``RZZ``.
    """
    T, n = paulis.shape[0], chain.n
    fx = np.zeros((T, n), dtype=bool)
    fz = np.zeros((T, n), dtype=bool)
    bond_signs = np.empty((T, depth, chain.bonds))
    mixer_signs = np.empty((T, depth, n))
    for layer in range(depth):
        for b in range(chain.bonds):
            bond_signs[:, layer, b] = 1 - 2 * (fx[:, b] ^ fx[:, b + 1])
            code = paulis[:, layer, b]
            fx[:, b] ^= (code & 1).astype(bool)
            fz[:, b] ^= (code >> 1 & 1).astype(bool)
            fx[:, b + 1] ^= (code >> 2 & 1).astype(bool)
            fz[:, b + 1] ^= (code >> 3 & 1).astype(bool)
        mixer_signs[:, layer] = 1 - 2 * fz
    flat = paulis.reshape(T, -1) != 0
    first = np.where(flat.any(axis=1), flat.argmax(axis=1) // chain.bonds, depth)
    return bond_signs, mixer_signs, fx, first


def _ideal_run(chain, depth, delta):
    """Noiseless evolution, keeping the state at the start of every layer."""
    gammas, betas = _schedule(depth, delta)
    psi = np.full((1, 2 ** chain.n), 2 ** (-chain.n / 2), dtype=complex)
    ones_b, ones_m = np.ones((1, chain.bonds)), np.ones((1, chain.n))
    starts = []
    for gamma, beta in zip(gammas, betas):
        starts.append(psi[0].copy())
        chain.layer(psi, gamma, beta, ones_b, ones_m)
    return psi, starts


def _simulate(chain, depth, delta, paulis, starts, batch):
    """``r`` and ``prob`` for every configuration in ``paulis``.

    A run is identical to the noiseless one until its first error, so it starts from the stored
    noiseless state of that layer. Runs are sorted by that layer, which makes the active ones a
    prefix of the batch.
    """
    gammas, betas = _schedule(depth, delta)
    bond_signs, mixer_signs, flips, first = _frames(chain, depth, paulis)
    order = np.argsort(first, kind="stable")
    bond_signs, mixer_signs, flips, first = (a[order] for a in (bond_signs, mixer_signs, flips, first))
    r_out, p_out = np.empty(len(paulis)), np.empty(len(paulis))
    for lo in range(0, len(paulis), batch):
        hi = min(lo + batch, len(paulis))
        psi = np.empty((hi - lo, 2 ** chain.n), dtype=complex)
        active = 0
        for layer in range(depth):
            now = int(np.searchsorted(first[lo:hi], layer, side="right"))
            psi[active:now] = starts[layer]
            active = now
            if active:
                chain.layer(psi[:active], gammas[layer], betas[layer],
                            bond_signs[lo:lo + active, layer], mixer_signs[lo:lo + active, layer])
        r_out[lo:hi], p_out[lo:hi] = chain.measure(psi, flips[lo:hi])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return r_out[inverse], p_out[inverse]


def _configurations(depth, bonds, k, count, rng, exhaustive):
    slots = depth * bonds
    if exhaustive:
        rows = [(pos, codes) for pos in itertools.combinations(range(slots), k)
                for codes in itertools.product(range(1, 16), repeat=k)]
        paulis = np.zeros((len(rows), slots), dtype=np.int8)
        for i, (pos, codes) in enumerate(rows):
            paulis[i, list(pos)] = codes
    else:
        pos = np.argsort(rng.random((count, slots)), axis=1)[:, :k]
        paulis = np.zeros((count, slots), dtype=np.int8)
        np.put_along_axis(paulis, pos, rng.integers(1, 16, size=(count, k), dtype=np.int8), axis=1)
    return paulis.reshape(-1, depth, bonds)


# ======================================================================================
# Conditional decay of one (N_q, p)
# ======================================================================================
@dataclass
class Decay:
    """``m_k`` for one chain length and depth, and every overlap that follows from it."""

    n: int
    depth: int
    delta: float
    r_ideal: float
    prob_ideal: float
    k: list = field(default_factory=list)
    r: list = field(default_factory=list)
    r_err: list = field(default_factory=list)
    prob: list = field(default_factory=list)
    prob_err: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    exact: list = field(default_factory=list)
    seconds: float = 0.0
    seed: int | None = None

    @property
    def edges(self) -> int:
        return self.n - 1

    @property
    def slots(self) -> int:
        """Edge interactions in the circuit, each a place where a fault can occur."""
        return self.depth * self.edges

    @property
    def r_rand(self) -> float:
        return 0.5

    @property
    def prob_rand(self) -> float:
        return 2 / 2 ** self.n

    def signal(self, observable="r"):
        """``(m_k - random) / (ideal - random)`` and its standard error, per ``k``."""
        ideal, rand = getattr(self, f"{observable}_ideal"), getattr(self, f"{observable}_rand")
        scale = ideal - rand
        return ((np.asarray(getattr(self, observable)) - rand) / scale,
                np.asarray(getattr(self, f"{observable}_err")) / abs(scale))

    def overlap(self, lam, observable="r"):
        """Overlap at every ``lambda``: ``(value, standard error, tail)``.

        ``tail`` is the probability of more errors than were simulated. That part counts as random,
        so a large tail means the overlap is underestimated, by at most ``tail`` times the last
        resolved signal.
        """
        from scipy.stats import binom

        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        q = slot_error_probability(lam)
        k = np.asarray(self.k)
        w = binom.pmf(k[None, :], self.slots, q[:, None])                  # (lambdas, k)
        s, e = self.signal(observable)
        tail = binom.sf(k.max(), self.slots, q)
        return w @ s, np.sqrt((w ** 2) @ (e ** 2)), tail

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def simulate_decay(n: int, depth: int, delta: float = 0.5, trajectories: int = 128,
                   exhaustive_limit: int | None = None, max_errors: int | None = None,
                   tolerance: float = 2.0, patience: int = 2, batch: int | None = None,
                   seed: int | None = None) -> Decay:
    """Estimate ``m_k``, the mean ``r`` and ``prob`` given exactly ``k`` faulty edge interactions.

    For each ``k``, every configuration is enumerated when there are at most ``exhaustive_limit``
    of them (default: ``trajectories``). Otherwise ``trajectories`` configurations are sampled. The
    loop ends at ``max_errors`` (default: every slot), or once both signals have stayed within
    ``tolerance`` standard errors of zero for ``patience`` consecutive ``k``.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    chain = _Chain(n)
    final, starts = _ideal_run(chain, depth, delta)
    r0, p0 = chain.measure(final, np.zeros((1, n), dtype=bool))
    out = Decay(n=n, depth=depth, delta=delta, r_ideal=float(r0[0]), prob_ideal=float(p0[0]),
                k=[0], r=[float(r0[0])], r_err=[0.0], prob=[float(p0[0])], prob_err=[0.0],
                runs=[1], exact=[True], seed=seed)
    limit = trajectories if exhaustive_limit is None else exhaustive_limit
    batch = batch or max(1, 2 ** 21 // 2 ** n)
    kmax = out.slots if max_errors is None else min(max_errors, out.slots)
    quiet = 0
    for k in range(1, kmax + 1):
        exhaustive = comb(out.slots, k) * 15 ** k <= limit
        paulis = _configurations(depth, n - 1, k, trajectories, rng, exhaustive)
        r, prob = _simulate(chain, depth, delta, paulis, starts, batch)
        size = len(r)
        out.k.append(k)
        out.runs.append(size)
        out.exact.append(bool(exhaustive))
        out.r.append(float(r.mean()))
        out.prob.append(float(prob.mean()))
        out.r_err.append(0.0 if exhaustive else float(r.std(ddof=1) / np.sqrt(size)))
        out.prob_err.append(0.0 if exhaustive else float(prob.std(ddof=1) / np.sqrt(size)))
        if not exhaustive:
            below = all(abs(s[-1]) < tolerance * e[-1] for s, e in map(out.signal, OBSERVABLES))
            quiet = quiet + 1 if below else 0
            if quiet >= patience:
                break
    out.seconds = time.perf_counter() - t0
    return out


def ideal_reference(n: int, depth: int, delta: float = 0.5):
    """Noiseless ``(r, probability of the optimum)`` of the chain at this depth."""
    chain = _Chain(n)
    final, _ = _ideal_run(chain, depth, delta)
    r, prob = chain.measure(final, np.zeros((1, n), dtype=bool))
    return float(r[0]), float(prob[0])


# ======================================================================================
# A study: many (N_q, p), cached on disk
# ======================================================================================
def run_study(nqs, depths, delta: float = 0.5, trajectories: int = 128, seed: int = 2026,
              cache: str | Path | None = None, recompute: bool = False, verbose: bool = True,
              **kwargs) -> dict:
    """``{(n, p): Decay}`` for every chain length and depth, reusing whatever ``cache`` holds.

    Tables in the cache are reused when their ``delta`` matches and they had at least
    ``trajectories`` runs per sampled ``k``. New ones are added and the file is rewritten.
    """
    cached = {} if (recompute or cache is None or not Path(cache).exists()) else load_study(cache)
    out = {}
    for n in nqs:
        for depth in depths:
            key = (int(n), int(depth))
            old = cached.get(key)
            usable = old is not None and old.delta == delta and min(
                (runs for runs, exact in zip(old.runs, old.exact) if not exact),
                default=trajectories) >= trajectories
            if usable:
                out[key] = old
                continue
            out[key] = simulate_decay(n, depth, delta, trajectories=trajectories,
                                      seed=None if seed is None else seed + 1000 * n + depth, **kwargs)
            d = out[key]
            if verbose:
                print(f"N_q={n:2d}  p={depth:2d}: k<={max(d.k):3d} of {d.slots:4d} edge interactions, "
                      f"{sum(d.runs):6d} runs, {d.seconds:6.1f} s")
            if cache is not None:
                save_study(cache, {**cached, **out})
    return out


def save_study(path, decays: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"model": "LR-QAOA 1D chain, depolarizing_error(lambda, 2) after every edge interaction",
            "note": "m_k does not depend on how lambda is defined",
            "tables": [d.to_dict() for _, d in sorted(decays.items())]}
    path.write_text(json.dumps(body, indent=1))
    return path


def load_study(path) -> dict:
    body = json.loads(Path(path).read_text())
    return {(d["n"], d["depth"]): Decay.from_dict(d) for d in body["tables"]}


# ======================================================================================
# Fits
# ======================================================================================
def collapse(decays: dict, n: int, lambdas, observable="r"):
    """All ``(eps_acc, overlap, error, depth, tail)`` points of one chain length, flattened."""
    rows = []
    lambdas = np.asarray(lambdas, dtype=float)
    for (nn, depth), d in sorted(decays.items()):
        if nn != n:
            continue
        value, err, tail = d.overlap(lambdas, observable)
        rows.append(np.stack([accumulated_error(d.edges, depth, lambdas), value, err,
                              np.full_like(lambdas, depth), tail]))
    if not rows:
        raise KeyError(f"no tables for N_q = {n}")
    return np.concatenate(rows, axis=1)


def fit_kappa(eps, overlap, max_tail=None, tail=None):
    """Least-squares ``kappa`` of ``r_ovl`` (or ``rho_ovl``) ``= 2^(-kappa eps)``, as in the earlier noise analysis.

    Points with a non-positive overlap are dropped, and so are points whose ``tail`` exceeds
    ``max_tail`` when both are given.
    """
    from scipy.optimize import curve_fit

    eps, overlap = np.asarray(eps, dtype=float), np.asarray(overlap, dtype=float)
    mask = np.isfinite(eps) & np.isfinite(overlap) & (overlap > 0)
    if max_tail is not None and tail is not None:
        mask &= np.asarray(tail) <= max_tail
    x, y = eps[mask], overlap[mask]
    popt, pcov = curve_fit(overlap_model, x, y, p0=[1.0], bounds=([0.0], [np.inf]))
    residuals = y - overlap_model(x, popt[0])
    return {"kappa": float(popt[0]), "stderr": float(np.sqrt(pcov[0, 0])),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "r2": float(1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)),
            "points": int(mask.sum())}


def fit_kappa0(nqs, kappas):
    """``kappa = kappa_0 / N_q`` by least squares, in closed form: ``(kappa_0, stderr, rss)``."""
    x = 1 / np.asarray(nqs, dtype=float)
    y = np.asarray(kappas, dtype=float)
    c = float(x @ y / (x @ x))
    rss = float(np.sum((y - c * x) ** 2))
    stderr = float(np.sqrt(rss / max(len(y) - 1, 1) / (x @ x)))
    return c, stderr, rss


def decaying_part(depths, overlaps, min_points: int = 3):
    """Drop the leading plateau of a measured overlap curve, keeping at least ``min_points``.

    At small ``p`` the overlap sits near 1 and can even rise with depth under shot noise, which a
    strictly decreasing model can only fit badly. Everything up to and including the maximum is
    dropped, unless that would leave fewer than ``min_points`` points (short scans are kept whole).
    """
    depths, overlaps = np.asarray(depths, dtype=float), np.asarray(overlaps, dtype=float)
    start = min(int(np.argmax(overlaps)) + 1, max(0, len(depths) - min_points))
    return depths[start:], overlaps[start:]


def fit_lambda_eff(depths, overlaps, edges: int, kappa: float):
    """Per-edge ``lambda_eff`` making ``2^(-kappa N_edges p lambda)`` match a device's overlaps."""
    from scipy.optimize import curve_fit

    p = np.asarray(depths, dtype=float)
    y = np.asarray(overlaps, dtype=float)

    def model(depth, lam):
        return overlap_model(accumulated_error(edges, depth, lam), kappa)

    popt, pcov = curve_fit(model, p, y, p0=[1e-3], bounds=([0.0], [1.0]))
    residuals = y - model(p, popt[0])
    return {"lambda_eff": float(popt[0]), "stderr": float(np.sqrt(pcov[0, 0])),
            "rmse": float(np.sqrt(np.mean(residuals ** 2)))}


# ======================================================================================
# Device runs
# ======================================================================================
def device_run(results_dir, backend: str, kind: str, stamps, n: int, depths=None):
    """``{p: (r, probability of the optimum)}`` of the ``n``-spin chain in the named run files
    ``<results_dir>/<backend>/chain/<kind>/<stamp>_<backend>_chain_<kind>.json``.

    Both numbers are recomputed from the stored samples (``analysis.load_results``). ``depths``
    keeps only those depths; depths that were not run are skipped.
    """
    from .analysis import load_results

    files = {f"{stamp}_{backend}_chain_{kind}.json" for stamp in stamps}
    out = {}
    for _, by_depth in load_results(results_dir, backend, kind=kind, n_data=n, files=files).items():
        for p, s in by_depth.items():
            if depths is None or p in depths:
                out[int(p)] = (s["r"], s["p_optimal"])
    return dict(sorted(out.items()))


def device_overlaps(run: dict, n: int, delta: float = 0.5, observable="r", cap: float | None = 1.2):
    """Depths and overlaps of a device run against the exact noiseless and random references.

    As in the reference analysis, overlaps above ``cap`` (shot noise on a nearly noiseless point)
    are set to 1.
    """
    depths = sorted(run)
    column = OBSERVABLES.index(observable)
    rand = 0.5 if observable == "r" else 2 / 2 ** n
    values = []
    for p in depths:
        ideal = ideal_reference(n, p, delta)[column]
        values.append((run[p][column] - rand) / (ideal - rand))
    values = np.asarray(values)
    if cap is not None:
        values[values > cap] = 1.0
    return np.asarray(depths), values
