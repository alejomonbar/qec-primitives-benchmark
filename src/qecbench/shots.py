"""Shot budgets: how many shots separate LR-QAOA from random guessing, or one device from another.

A benchmark result is the mean of ``S`` single-shot values of ``r``. Random guessing gives ``r_rand``
with shot-noise ``sigma_rand(S)`` (exact, from ``lrqaoa.random_baseline``), and a run is told apart from
it when its mean clears ``r_rand + n_sigma * sigma_rand(S)``. ``separation_probability`` is how often that
happens at each ``S`` for a single-shot distribution: the noiseless one (``ideal_r_distribution``) gives
the least budget any device needs, a measured one (``sample_distribution``) what that run needed.

The run's mean is resampled ``n_boot`` times with a seeded generator, so a figure is reproducible.

Ranking two devices is the same question between two distributions: ``shots_to_rank`` is the number of
shots per device at which their means differ by ``z`` standard errors. ``random_r_distribution`` and
``white_noise_distribution`` give the distributions of a device modelled as a mixture of the noiseless
state and uniform noise.
"""

from __future__ import annotations

import numpy as np

from .lrqaoa import bitstring_energies, energies, ideal_energy_distribution, ideal_r, random_baseline


def ideal_r_distribution(primitive, depth: int, delta: float = 0.5):
    """Single-shot distribution of ``r`` on a noiseless device: ``(values, probabilities)``.

    Exact up to ``lrqaoa.MAX_STATEVECTOR_QUBITS`` data qubits, beyond that a stored distribution (see
    ``lrqaoa.ideal_energy_distribution``).
    """
    levels, probabilities = ideal_energy_distribution(primitive.hamiltonian, primitive.n_data, depth, delta)
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    return (levels - e_max) / (e_opt - e_max), probabilities


def sample_distribution(counts, primitive, depth: int | None = None, delta: float = 0.5, quantity: str = "r"):
    """Single-shot distribution of ``r`` (or ``r_ovl``, which needs ``depth``) in measured counts."""
    bitstrings = list(counts)
    weights = np.array([counts[b] for b in bitstrings], dtype=float)
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    values = (bitstring_energies(bitstrings, primitive.hamiltonian) - e_max) / (e_opt - e_max)
    if quantity == "r_ovl":
        if depth is None:
            raise ValueError("r_ovl needs the depth, for its noiseless reference")
        r_rand = random_baseline(primitive, 1)[0]
        values = (values - r_rand) / (ideal_r(primitive, depth, delta) - r_rand)
    elif quantity != "r":
        raise ValueError(f"quantity is 'r' or 'r_ovl', not {quantity!r}")
    return values, weights / weights.sum()


def resampled_means(values, probabilities, shots: int, n_boot: int, rng) -> np.ndarray:
    """``n_boot`` means of ``shots`` draws from the distribution."""
    draws = rng.choice(len(values), size=(n_boot, int(shots)), p=probabilities)
    return np.asarray(values)[draws].mean(axis=1)


def separation_probability(primitive, values, probabilities, shots_grid, n_sigma: float = 3.0,
                           n_boot: int = 20_000, seed: int = 0) -> np.ndarray:
    """``P(mean of S shots > r_rand + n_sigma * sigma_rand(S))`` for each ``S`` in ``shots_grid``.

    The random-guessing mean and spread are exact (``random_baseline``); only the distribution of the
    run's mean is resampled.
    """
    rng = np.random.default_rng(seed)
    out = []
    for shots in shots_grid:
        r_rand, sigma = random_baseline(primitive, int(shots))
        out.append(float(np.mean(resampled_means(values, probabilities, shots, n_boot, rng) > r_rand + n_sigma * sigma)))
    return np.array(out)


def shots_to_separate(shots_grid, probabilities, level: float = 0.95):
    """The first ``S`` whose separation probability reaches ``level``, or None."""
    return next((int(s) for s, q in zip(shots_grid, probabilities) if q >= level), None)


def random_r_distribution(primitive):
    """Exact single-shot distribution of ``r`` for uniformly random bitstrings, on the levels of
    ``ideal_r_distribution`` (all energy levels of the Hamiltonian, up to 26 data qubits)."""
    table = energies(primitive.hamiltonian, primitive.n_data)
    levels, counts = np.unique(np.round(table, 9), return_counts=True)
    e_opt, e_max = primitive.optimal_energy(), primitive.max_energy()
    return (levels - e_max) / (e_opt - e_max), counts / counts.sum()


def white_noise_distribution(ideal_probabilities, random_probabilities, attenuation: float):
    """``A P_ideal + (1 - A) P_random``: a device that keeps the noiseless state with weight ``A``."""
    return attenuation * np.asarray(ideal_probabilities) + (1 - attenuation) * np.asarray(random_probabilities)


def moments(values, probabilities):
    """Mean and standard deviation of a single-shot distribution."""
    mean = float(np.dot(values, probabilities))
    return mean, float(np.sqrt(np.dot(probabilities, (np.asarray(values) - mean) ** 2)))


def shots_to_rank(mean_worse, std_worse, mean_better, std_better, z: float = 3.0) -> float:
    """Shots per device for the difference of two means to reach ``z`` standard errors:
    ``z^2 (std_worse^2 + std_better^2) / (mean_better - mean_worse)^2``."""
    return z ** 2 * (std_worse ** 2 + std_better ** 2) / (mean_better - mean_worse) ** 2
