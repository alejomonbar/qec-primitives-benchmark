"""A simple, device-independent noise model: uniform depolarizing plus readout error.

This is what ``SimBackend`` runs, so the whole benchmark works with no vendor account at all.
It deliberately does **not** imitate a particular chip - ``IBMBackend(..., local=True)`` does
that, from the device's own calibration.  Here every qubit and every coupler is equally noisy,
which makes it useful for checking the pipeline, for teaching, and for asking "what would this
benchmark look like at a given error rate?".

The defaults are the order of magnitude of a good superconducting device in 2026:

====================  =========  ==================================================
``p_1q``              2.5e-4     depolarizing probability per single-qubit gate
``p_2q``              4.0e-3     depolarizing probability per two-qubit gate
``readout``           1.2e-2     probability a measured bit comes back flipped
====================  =========  ==================================================

They are parameters, not knobs in the notebook: pass them when you want to explore, e.g.
``SimBackend(p_2q=0.01)`` or ``SimBackend(readout=0, p_1q=0)``.
"""

from __future__ import annotations

DEFAULTS = {"p_1q": 2.5e-4, "p_2q": 4.0e-3, "readout": 1.2e-2}

# rz is a frame change on real hardware (no pulse, no error); the rest are actual gates.
ONE_QUBIT_GATES = ("x", "sx", "h", "rx", "ry", "r", "u")
TWO_QUBIT_GATES = ("cz", "cx", "ecr", "rzz")


def depolarizing_noise_model(p_1q=DEFAULTS["p_1q"], p_2q=DEFAULTS["p_2q"],
                             readout=DEFAULTS["readout"], one_qubit_gates=ONE_QUBIT_GATES,
                             two_qubit_gates=TWO_QUBIT_GATES):
    """An Aer noise model with the same error on every qubit and coupler."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    model = NoiseModel(basis_gates=list(one_qubit_gates) + list(two_qubit_gates) + ["rz"])
    if p_1q:
        model.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), list(one_qubit_gates))
    if p_2q:
        model.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), list(two_qubit_gates))
    if readout:
        model.add_all_qubit_readout_error(
            ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return model


def describe(p_1q=DEFAULTS["p_1q"], p_2q=DEFAULTS["p_2q"], readout=DEFAULTS["readout"]) -> str:
    """One line naming the noise, stored with every result file produced under it."""
    return (f"uniform depolarizing noise: {p_1q:.2e} per 1q gate, {p_2q:.2e} per 2q gate, "
            f"{readout:.2e} readout error")
