"""qecbench - hardware benchmarks of QEC primitives built on mid-circuit measurement.

Layers, bottom up::

    primitives   what is benchmarked: Chain (measurement gadget) and Direct (the same Ising
                 chain run directly, no ancilla); surface/colour code patches later
    lrqaoa       schedule, energies, exact noiseless and random references
    layout       where it goes on a chip: selection, packing, validation
    circuits     the gadget in two dialects (Qiskit dynamic, IQM natives)
    backends     vendor adapters: AerBackend, IBMBackend, IQMBackend
    experiment   plan -> submit (manifest) -> harvest
    analysis     counts -> r, r_ovl; result files (legacy compatible)
    plotting     device maps, partitions, depth sweeps
"""

from .primitives import Chain, Direct, Primitive, Term, from_dict

__version__ = "0.1.0"
__all__ = ["Chain", "Direct", "Primitive", "Term", "from_dict"]
