"""The vendor interface every backend adapter implements.

The experiment layer (plan -> manifest -> harvest) only talks to this class, so a new
vendor is one adapter, and a new primitive family needs no adapter changes at all.
"""

from __future__ import annotations

from ..layout import validate_batch


class Backend:
    vendor = "generic"
    kinds: tuple[str, ...] = ("mcm",)
    is_simulator = False
    max_circuits_per_job = 1

    def __init__(self, name: str):
        self.name = name

    @property
    def simulated(self) -> bool:
        """True when results come from a simulator rather than from the device itself."""
        return bool(self.is_simulator or getattr(self, "local", False))

    @property
    def noise_description(self) -> str | None:
        """What the simulation's noise came from; recorded with the results."""
        return None

    # -- device ---------------------------------------------------------------------------
    def coupling_graph(self):
        raise NotImplementedError

    def feedforward_groups(self):
        """``{group: [qubits]}`` when feed-forward may not cross groups (IQM), else None."""
        return None

    def calibration(self, save_dir=None):
        return None

    def error_budget(self, instance, cal) -> float | None:
        """First-order error per LR-QAOA layer from the vendor calibration (None if unknown)."""
        return None

    def flag_instances(self, instances, cal, **thresholds) -> list[tuple]:
        """``[(instance, reason)]`` for instances sitting on hardware outside the thresholds.

        Every chip has a few dead couplers and unreadable qubits; an instance touching one
        measures that defect rather than the primitive, so it is worth knowing before
        submitting.  A vendor without calibration data reports nothing.
        """
        return []

    # -- circuits -------------------------------------------------------------------------
    def validate(self, batch, kind="mcm") -> list[str]:
        """Everything that would make the device reject or mis-execute this circuit."""
        return validate_batch(batch, self.coupling_graph())

    def build(self, batch, depth, delta, kind="mcm"):
        raise NotImplementedError

    def check_built(self, circuit, batch) -> list[str]:
        """Problems visible only after compilation (e.g. routing inserted by a transpiler)."""
        return []

    def estimate(self, plan) -> dict | None:
        """``{"unit", "per_circuit", "total", "usd", "notes"}`` for the plan's circuits."""
        return None

    # -- execution ------------------------------------------------------------------------
    def submit(self, circuits, shots) -> dict:
        """Send one job; return the fields to store in its manifest record (``job_id``...)."""
        raise NotImplementedError

    def fetch(self, record, tasks):
        """Counts of a submitted job, or None if it is not finished.

        Returns one list per task (= circuit) of the job, holding one counts dict per instance
        of that circuit, keyed by data bitstrings read left to right (data qubit 0 first).
        """
        raise NotImplementedError


def reverse_keys(counts):
    """Qiskit register strings are little-endian; flip them to data-qubit order."""
    return {bits[::-1]: int(n) for bits, n in counts.items()}
